import os
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from einops import rearrange
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer
import math

from transformers import AutoTokenizer, AutoModelForCausalLM

from eeg_encoders.cbramod import CBraMod_Wrapper
from eeg_encoders.labram import LaBraM
from eeg_encoders.tfm_tokenizer import TFM_Tokenizer

from utils.utils import seed_everything


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 12000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        if x.size(1) > self.pe.size(1):
            x = x[:, : self.pe.size(1)]  # truncate the input if it is longer than the positional encoding
        x = x + self.pe[:, : x.size(1)] 
        return self.dropout(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 emb_size = 64,
                 num_heads = 8,
                 depth = 4,
                 max_seq_len = 1024,   
                 ):
        super().__init__()
        
        self.transformer = LinearAttentionTransformer(
            dim = emb_size,
            heads = num_heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )  
        
    def forward(self, x):
        x = self.transformer(x)
        return x
    
class PerceiverBlock(nn.Module):
    """Single Perceiver-style block."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        
        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, queries, context, context_mask=None):
        # Cross-attention
        q_norm = self.cross_attn_norm_q(queries)
        kv_norm = self.cross_attn_norm_kv(context)
        
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = context_mask.squeeze(1).squeeze(1) < -1e8
        
        attn_out, _ = self.cross_attn(
            q_norm, kv_norm, kv_norm, key_padding_mask=key_padding_mask
        )
        queries = queries + attn_out
        
        # Feed-forward
        queries = queries + self.ff(self.ff_norm(queries))
        
        return queries
    
# ============================================================================
# Projection Layers
# ============================================================================

class LinearProjector(nn.Module):
    """Simple linear projection"""
    def __init__(self, encoder_dim: int, llm_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.proj = nn.Linear(encoder_dim, llm_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class SequenceTransformerLinearProjector(nn.Module):
    """Sequence Transformer Linear Projector"""
    def __init__(self, encoder_dim: int, llm_dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        
        self.pos_enc = PositionalEncoding(d_model=encoder_dim, dropout=0.1, max_len= 12000) #12000)
        self.transformer = TransformerEncoder(emb_size=encoder_dim, num_heads=num_heads, depth=num_layers, max_seq_len= 12000) #12000)
        self.proj = nn.Linear(encoder_dim, llm_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.squeeze(0)
        return self.proj(x)
    

class PerceiverProjector(nn.Module):
    """
    Perceiver-style compression of EEG embeddings to LLM embedding space
    """
    
    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_queries: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, encoder_dim) * 0.02)
        self.perceiver_blocks = nn.ModuleList([PerceiverBlock(dim=encoder_dim, num_heads=num_heads, dropout=dropout, ff_mult=2) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(encoder_dim)
        self.proj = nn.Linear(encoder_dim, llm_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        queries = self.query_tokens.expand(x.shape[0], -1, -1)
        
        for block in self.perceiver_blocks:
            queries = block(queries, x)
            
        queries = self.final_norm(queries)
        queries = self.proj(queries)

        if queries.ndim >= 2:
            queries = queries.squeeze(0)
        return queries
    
class SequencePerceiverProjector(nn.Module):
    """
    Sequence Transformer then Perceiver-style compression of EEG embeddings to LLM embedding space
    """
    
    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_queries: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        
        self.pos_enc = PositionalEncoding(d_model=encoder_dim, dropout=0.1, max_len=12000)
        self.transformer = TransformerEncoder(emb_size=encoder_dim, num_heads=num_heads, depth=num_layers, max_seq_len=12000)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, encoder_dim) * 0.02)
        self.perceiver_blocks = nn.ModuleList([PerceiverBlock(dim=encoder_dim, num_heads=num_heads, dropout=dropout, ff_mult=2) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(encoder_dim)
        self.proj = nn.Linear(encoder_dim, llm_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.pos_enc(x)
        x = self.transformer(x)
        
        queries = self.query_tokens.expand(x.shape[0], -1, -1)
        
        for block in self.perceiver_blocks:
            queries = block(queries, x)
            
        queries = self.final_norm(queries)
        queries = self.proj(queries)

        if queries.ndim >= 2:
            queries = queries.squeeze(0)
        return queries

# ============================================================================
# EEG Encoder
# ============================================================================
            
class EEGEncoder(nn.Module):
    def __init__(self, 
                 encoder_type: str = "cbramod",
                 device: str = "cuda:0",                 
                 aggregation_method: str = "cls",# "cls", "mean", "max", 'cls_then_attention',
                 llm_embedding_dim: int = 768,
                 ): 
        super().__init__()
        self.cls_only = 'True' if 'cls' in aggregation_method else 'False'
        self.aggregation_method = aggregation_method
        # self.device = device
        self.llm_embedding_dim = llm_embedding_dim
        
        # Initialize the EEG Encoder
        if encoder_type == "cbramod":
            self.eeg_encoder = CBraMod_Wrapper(device=device,cls_only=self.cls_only)
        elif encoder_type == "labram":
            self.eeg_encoder = LaBraM(device=device,cls_only=self.cls_only)
        elif encoder_type == "tfm_tokenizer":
            self.eeg_encoder = TFM_Tokenizer(device=device,cls_only=self.cls_only)
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")

        # Create a EEG Padding Embedding
        # self.eeg_padding_embedding = nn.Embedding(1, self.config.encoder_hidden_dim)
        # nn.init.zeros_(self.eeg_padding_embedding.weight)
        # self.eeg_padding_embedding.weight.data[0, :] = 0.0
    @property
    def device(self):
        """Get actual device from model parameters."""
        try:
            return next(self.eeg_encoder.parameters()).device
        except StopIteration:
            return torch.device('cpu')
        
    def forward(self, x: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        x: List of eeg samples (batch), each eeg sample is a list of eeg sessions, 
            where each eeg session is a tensor of shape (S,C, T) 
            # S is the number of segments, 
            # C is the number of channels, 
            # T is the number of samples
        """
        current_device = self.device
        eeg_embed = []
        for eeg_sample in x:
            eeg_sample_embed = []
            for eeg_session in eeg_sample:
                # print(eeg_session.shape)
                if eeg_session.shape[0] > 500:
                    # print(f'batching eeg session to get embedding as it is too long {eeg_session.shape[0]}')
                    eeg_session_embed = []
                    for i in range(0, eeg_session.shape[0], 500):
                        # if eeg_session[i:i+500].ndim == 1:
                        #     eeg_session_embed.append(self.eeg_encoder(eeg_session[i:i+500].unsqueeze(0).to(current_device)))
                        # else:
                        eeg_session_embed.append(self.eeg_encoder(eeg_session[i:i+500].to(current_device)))
                        if eeg_session_embed[-1].ndim == 1:
                            print(f'eeg_session_embed[-1].ndim: {eeg_session_embed[-1].ndim}')
                            eeg_session_embed[-1] = eeg_session_embed[-1].unsqueeze(0)
                    
                    eeg_session_embed = torch.cat(eeg_session_embed, dim=0)
                    # print(f'eeg_session_embed.shape: {eeg_session_embed.shape}')
                else:
                    eeg_session_embed = self.eeg_encoder(eeg_session.to(current_device))
                eeg_sample_embed.append(eeg_session_embed)
            eeg_embed.append(eeg_sample_embed)
            
        return eeg_embed
    
    
class EEGLLM(nn.Module):
    def __init__(self,
                 eeg_encoder: Optional[nn.Module] = None,
                 llm: Optional[nn.Module] = None,
                 tokenizer = None,
                 projector: Optional[nn.Module] = None,
                 training_mode: str = "projection_only", # "projection_only", "projection_then_lora"
                 use_chat_template: bool = True,
                ):
        super().__init__()
        
        
        self.eeg_encoder = eeg_encoder
        # self.device = self.eeg_encoder.device
        self.llm = llm
        self.tokenizer = tokenizer
        self.projector = projector
        self.training_mode = training_mode
        self.use_chat_template = use_chat_template
        
        self.llm_embedding_dim = self.projector.llm_dim
        
        
        # Create Embedding for Special Tokens
        self.start_eeg_token = nn.Parameter(torch.zeros(1, self.llm_embedding_dim))
        self.end_eeg_token = nn.Parameter(torch.zeros(1, self.llm_embedding_dim))
        self.eeg_session_separator_token = nn.Parameter(torch.zeros(1, self.llm_embedding_dim))
        
        # Initialize the Special Tokens
        nn.init.xavier_uniform_(self.start_eeg_token)
        nn.init.xavier_uniform_(self.end_eeg_token)
        nn.init.xavier_uniform_(self.eeg_session_separator_token)
        
        self._freeze_module(self.eeg_encoder)
        if self.training_mode != "projection_then_lora":
            self._freeze_module(self.llm)
            
            
    @property
    def device(self):
        """Get actual device from model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
        
        
    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters."""
        for param in module.parameters():
            param.requires_grad = False

    def encode_eeg(self, x: List[List[torch.Tensor]]) -> torch.Tensor:
        with torch.no_grad():
            eeg_embeddings = self.eeg_encoder(x)
            
        # Project the EEG embeddings to llm embedding space
        if self.projector is not None:
            eeg_tokens = []
            for eeg_sample in eeg_embeddings:
                eeg_sample_tokens = []
                for eeg_session in eeg_sample:
                    eeg_session_tokens = self.projector(eeg_session)
                    eeg_session_tokens = torch.cat([eeg_session_tokens, self.eeg_session_separator_token], dim=0)
                    eeg_sample_tokens.append(eeg_session_tokens)
                eeg_sample_tokens = torch.cat(eeg_sample_tokens, dim=0)
                eeg_tokens.append(eeg_sample_tokens)
        else:
            raise ValueError(f"Projector is not set for {self.training_mode} training mode")
            
        return eeg_tokens
    
    def format_prompt_for_chat(self, prompt: str, response: str = None) -> Tuple[str, str]:
        """
        Format prompt using chat template for instruction-tuned models.
        Returns formatted prompt and the assistant prefix to identify where response starts.
        """
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            # Create conversation format
            messages = [{"role": "user", "content": prompt}]
            
            # Get the formatted prompt (without assistant response)
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True  # Adds assistant turn marker
            )
            return formatted_prompt
        else:
            # For non-instruction-tuned models, return as-is
            return prompt
    
    
    def prepare_multimodal_inputs(self, 
                eeg_data: List[List[torch.Tensor]], 
                prompts: List[str],
                labels: List[str] = None,
                ) -> Dict[str, torch.Tensor]:
        
        
        batch_size = len(prompts)
        
        # Get current device dynamically
        current_device = self.device
        
        # Encode EEG data
        eeg_tokens_list = self.encode_eeg(eeg_data)
        
        all_input_embeds = []
        all_attention_masks = []
        all_labels = []
        
        
        for i in range(batch_size):
            # -----------------------------------------------------------------
            # 1. Get EEG embeddings for this sample
            # -----------------------------------------------------------------
            eeg_embeds = eeg_tokens_list[i]  # Shape: (num_eeg_tokens, llm_dim)
            if eeg_embeds.ndim == 2:
                eeg_embeds = eeg_embeds.unsqueeze(0)  # (1, num_eeg_tokens, llm_dim)
                
            # Add start and end tokens
            start_token = self.start_eeg_token.unsqueeze(0)  # (1, 1, llm_dim)
            end_token = self.end_eeg_token.unsqueeze(0)      # (1, 1, llm_dim)
            eeg_embeds_with_special = torch.cat([start_token, eeg_embeds, end_token], dim=1)
            num_eeg_tokens = eeg_embeds_with_special.shape[1]
            
            # print(f'eeg_embeds.shape: {eeg_embeds.shape}')
            # print(f'eeg_embeds_with_special.shape: {eeg_embeds_with_special.shape}')
            # print(f'num_eeg_tokens: {num_eeg_tokens}')
    
            # -----------------------------------------------------------------
            # 2. Format and tokenize the prompt
            # -----------------------------------------------------------------
            formatted_prompt = self.format_prompt_for_chat(prompts[i])
            prompt_encoding = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                add_special_tokens=True
            )
            prompt_ids = prompt_encoding.input_ids.to(current_device)
            prompt_attention_mask = prompt_encoding.attention_mask.to(current_device)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
            num_prompt_tokens = prompt_ids.shape[1]
            # print(f'prompt_ids.shape: {prompt_ids.shape}')
            # print(f'prompt_attention_mask.shape: {prompt_attention_mask.shape}')
            # print(f'prompt_embeds.shape: {prompt_embeds.shape}')
            # print(f'num_prompt_tokens: {num_prompt_tokens}')
            # print(num_eeg_tokens, num_prompt_tokens)
            # -----------------------------------------------------------------
            # 3. Process labels (response) if provided
            # -----------------------------------------------------------------
            if labels is not None and labels[i] is not None:
                # Tokenize the response (the target JSON report)
                response_encoding = self.tokenizer(
                    labels[i],
                    return_tensors="pt",
                    add_special_tokens=False  # Don't add BOS again
                )
                response_ids = response_encoding.input_ids.to(current_device)
                response_attention_mask = response_encoding.attention_mask.to(current_device)
                response_embeds = self.llm.get_input_embeddings()(response_ids)
                num_response_tokens = response_ids.shape[1]
                # print(f'response_ids.shape: {response_ids.shape}')
                # print(f'response_attention_mask.shape: {response_attention_mask.shape}')
                # print(f'response_embeds.shape: {response_embeds.shape}')
                # print(f'num_response_tokens: {num_response_tokens}')
                
                # Add EOS token to response
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None:
                    eos_ids = torch.tensor([[eos_token_id]], device=current_device)
                    eos_embeds = self.llm.get_input_embeddings()(eos_ids)
                    response_ids = torch.cat([response_ids, eos_ids], dim=1)
                    response_embeds = torch.cat([response_embeds, eos_embeds], dim=1)
                    response_attention_mask = torch.cat([
                        response_attention_mask, 
                        torch.ones((1, 1), device=current_device, dtype=response_attention_mask.dtype)
                    ], dim=1)
                    num_response_tokens += 1
                
                
                # -----------------------------------------------------------------
                # 4. Combine embeddings to get multimodal embeddings: [EEG] + [Prompt] + [Response]
                # -----------------------------------------------------------------
                combined_embeds = torch.cat([
                        eeg_embeds_with_special,  # (1, num_eeg, dim)
                        prompt_embeds,             # (1, num_prompt, dim)
                        response_embeds            # (1, num_response, dim)
                    ], dim=1)
            
                # -----------------------------------------------------------------
                # 5. Combine attention masks
                # -----------------------------------------------------------------
                eeg_attention_mask = torch.ones(
                        (1, num_eeg_tokens), 
                        dtype=prompt_attention_mask.dtype, 
                        device=current_device
                    )
                
                combined_attention_mask = torch.cat([
                        eeg_attention_mask,
                        prompt_attention_mask,
                        response_attention_mask
                        ], dim=1)
                
                # -----------------------------------------------------------------
                # 6. Create labels with masking for SFT
                # -----------------------------------------------------------------

                # Create ignore mask for EEG and prompt tokens
                num_input_tokens = num_eeg_tokens + num_prompt_tokens
                input_mask = torch.full(
                    (1, num_input_tokens),
                    fill_value=-100,  # -100 is ignored by CrossEntropyLoss
                    dtype=torch.long,
                    device=current_device
                )
                # print(f'input_mask.shape: {input_mask.shape}')
        
                # Labels structure: [-100, -100, ..., -100, response_token_1, response_token_2, ..., eos]
                # The model predicts token[i+1] given tokens[0:i+1] --> NFT task
                # loss only on predicting response tokens
                combined_labels = torch.cat([input_mask, response_ids], dim=1)
                # print(f'combined_labels.shape: {combined_labels.shape}')
                
                all_input_embeds.append(combined_embeds)
                all_attention_masks.append(combined_attention_mask)
                all_labels.append(combined_labels)
                

            else:
                # -----------------------------------------------------------------
                # Inference mode
                # -----------------------------------------------------------------
                
                # -----------------------------------------------------------------
                # 4. Combine embeddings to get multimodal embeddings: [EEG] + [Prompt]
                # -----------------------------------------------------------------
                combined_embeds = torch.cat([
                    eeg_embeds_with_special,
                    prompt_embeds
                ], dim=1)

                eeg_attention_mask = torch.ones(
                    (1, num_eeg_tokens), 
                    dtype=prompt_attention_mask.dtype, 
                    device=current_device
                )
                combined_attention_mask = torch.cat([
                    eeg_attention_mask,
                    prompt_attention_mask
                ], dim=1)
                
                all_input_embeds.append(combined_embeds)
                all_attention_masks.append(combined_attention_mask)
                all_labels.append(None)
        
        #-----------------------------------------------------------------
        # 7. Pad sequences to same length within batch
        # -----------------------------------------------------------------
        return self._pad_batch(all_input_embeds, all_attention_masks, all_labels)
    
    def _pad_batch(
        self, 
        embeds_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        labels_list: List[Optional[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Pad all sequences in batch to the same length.
        """
        # Find max length
        max_len = max(e.shape[1] for e in embeds_list)
        
        batch_embeds = []
        batch_attention_mask = []
        batch_labels = []
        
        for i in range(len(embeds_list)):
            embeds = embeds_list[i]
            attention_mask = attention_mask_list[i]
            labels = labels_list[i]
            
            seq_len = embeds.shape[1]
            pad_len = max_len - seq_len
            
            if pad_len > 0:
                # Pad embeddings with zeros
                embed_padding = torch.zeros(
                    (1, pad_len, embeds.shape[2]),
                    dtype=embeds.dtype,
                    device=embeds.device
                )
                embeds = torch.cat([embeds, embed_padding], dim=1)
                
                # Pad attention mask with zeros (don't attend to padding)
                attn_padding = torch.zeros(
                    (1, pad_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, attn_padding], dim=1)
                
                # Pad labels with -100 (ignore padding in loss)
                if labels is not None:
                    label_padding = torch.full(
                        (1, pad_len),
                        fill_value=-100,
                        dtype=labels.dtype,
                        device=labels.device
                    )
                    labels = torch.cat([labels, label_padding], dim=1)
            
            batch_embeds.append(embeds)
            batch_attention_mask.append(attention_mask)
            if labels is not None:
                batch_labels.append(labels)
        
        # Stack into batch tensors
        result = {
            "inputs_embeds": torch.cat(batch_embeds, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
        }
        
        if batch_labels:
            result["labels"] = torch.cat(batch_labels, dim=0)
        
        return result
    
    def forward(
        self,
        eeg_data: List[List[torch.Tensor]],
        prompts: List[str],
        labels: List[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            eeg_data: List of EEG data
            prompts: List of prompts
            labels: List of target responses (for training)
            
        Returns:
            Dictionary with loss (if training) and logits
        """
        
        # Prepare multimodal inputs
        model_inputs = self.prepare_multimodal_inputs(eeg_data, prompts, labels)
        
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs.get("labels", None),
            return_dict=True,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        eeg_data: List[List[torch.Tensor]],
        prompts: List[str],
        max_new_tokens: int = 2048,
        **kwargs
    ) -> List[str]:
        
        """
        Generate reports given EEG data and prompts.
        """
        self.eval()
        
        # Prepare inputs (no labels for generation)
        model_inputs = self.prepare_multimodal_inputs(eeg_data, prompts, labels=None)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # return_full_text=False,
            **kwargs
        )
        
        
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
    
    

def create_eeg_llm(
    config: Dict[str, Any],
    device: str = "cuda:0",
    checkpoint_path: str = None,
    load_projector_only: bool = False,
    **kwargs
) -> EEGLLM:
    """
    Create an EEGLLM model.
    """
    # Convert device string to handle accelerate devices properly
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device
    
    
    eeg_encoder = EEGEncoder(
        encoder_type=config['eeg_encoder_model'],
        device=device_str,
        aggregation_method=config['eeg_aggregation_method']
    )
    eeg_encoder = eeg_encoder.to(device_str)  # Then move to target device
    # eeg_encoder.device = device_str  
    
    tokenizer = AutoTokenizer.from_pretrained(config['llm_model'],device=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    llm = AutoModelForCausalLM.from_pretrained(config['llm_model'],
                                               attn_implementation=config['attn_implementation']).to(device)
    
    if config['projector'] == 'linear':
        projector = LinearProjector(
            encoder_dim=config['eeg_encoder_dim'],
            llm_dim=config['llm_dim']
        ).to(device)
    elif config['projector'] == 'sequence_transformer_linear_projector':
        projector = SequenceTransformerLinearProjector(
            encoder_dim=config['eeg_encoder_dim'],
            llm_dim=config['llm_dim']
        ).to(device)
    elif config['projector'] == 'perceiver_projector':
        projector = PerceiverProjector(
            encoder_dim=config['eeg_encoder_dim'],
            llm_dim=config['llm_dim']
        ).to(device)
    elif config['projector'] == 'sequence_transformer_perceiver_projector':
        projector = SequencePerceiverProjector(
            encoder_dim=config['eeg_encoder_dim'],
            llm_dim=config['llm_dim']
        ).to(device)
    else:
        raise ValueError(f"Invalid projector type: {config['projector']}")
    
    # check if the projection llm dim is the same as the llm embedding dim
    if projector.llm_dim != llm.get_input_embeddings().weight.shape[1]:
        raise ValueError(f"Projector llm dim {projector.llm_dim} is not the same as the llm embedding dim {llm.get_input_embeddings().weight.shape[1]}")
    

    # create the eeg llm
    eeg_llm = EEGLLM(
        eeg_encoder=eeg_encoder,
        llm=llm,
        tokenizer=tokenizer,
        projector=projector,
        training_mode=config['training_mode'],
        use_chat_template=config['use_chat_template']
    ).to(device)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if load_projector_only:
            # Load only projector and special tokens
            if 'projector_state_dict' in checkpoint:
                eeg_llm.projector.load_state_dict(checkpoint['projector_state_dict'])
                print("  ✓ Loaded projector state dict")
            
            if 'start_eeg_token' in checkpoint:
                eeg_llm.start_eeg_token.data = checkpoint['start_eeg_token'].to(device)
                print("  ✓ Loaded start_eeg_token")
            
            if 'end_eeg_token' in checkpoint:
                eeg_llm.end_eeg_token.data = checkpoint['end_eeg_token'].to(device)
                print("  ✓ Loaded end_eeg_token")
            
            if 'eeg_session_separator_token' in checkpoint:
                eeg_llm.eeg_session_separator_token.data = checkpoint['eeg_session_separator_token'].to(device)
                print("  ✓ Loaded eeg_session_separator_token")
        else:
            # Attempt to load full model state dict
            if 'model_state_dict' in checkpoint:
                eeg_llm.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  ✓ Loaded full model state dict")
            else:
                # Fallback to loading individual components
                if 'projector_state_dict' in checkpoint:
                    eeg_llm.projector.load_state_dict(checkpoint['projector_state_dict'])
                if 'start_eeg_token' in checkpoint:
                    eeg_llm.start_eeg_token.data = checkpoint['start_eeg_token'].to(device)
                if 'end_eeg_token' in checkpoint:
                    eeg_llm.end_eeg_token.data = checkpoint['end_eeg_token'].to(device)
                if 'eeg_session_separator_token' in checkpoint:
                    eeg_llm.eeg_session_separator_token.data = checkpoint['eeg_session_separator_token'].to(device)
                print("  ✓ Loaded projector and special tokens (fallback)")
        
        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"  Checkpoint from step: {checkpoint['step']}")
        if 'loss' in checkpoint:
            print(f"  Checkpoint loss: {checkpoint['loss']:.4f}")
    

    return eeg_llm

