from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from peft import PeftModel

_model = None
_tokenizer = None
_pipeline = None

_finetuned_model = None
_finetuned_tokenizer = None
_finetuned_pipeline = None

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def get_llm():
    global _model, _tokenizer, _pipeline
    
    if _model is None or _tokenizer is None:
        print("Initializing model")
        model_id = "microsoft/Phi-4-mini-instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(model_id)
    
    if _pipeline is None:
        _pipeline = pipeline("text-generation", model=_model, tokenizer=_tokenizer, max_new_tokens=1024, return_full_text=False)
        print("Pipeline initialized.")
    
    hf_pipeline = HuggingFacePipeline(pipeline=_pipeline)
    return ChatHuggingFace(llm=hf_pipeline)


def get_finetuned_llm():
    """파인튜닝된 LoRA 모델을 로드하여 LLM을 반환합니다."""
    global _finetuned_model, _finetuned_tokenizer, _finetuned_pipeline
    
    if _finetuned_model is None or _finetuned_tokenizer is None:
        print("Initializing finetuned model")
        base_model_id = "microsoft/Phi-4-mini-instruct"
        lora_path = "/Users/yhep/DRC/KEnK/Integral/Finetuning/lora_phi4mini_mps"
        
        try:
            # 기본 모델과 토크나이저 로드
            print("Loading base model and tokenizer...")
            _finetuned_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
            
            # LoRA 어댑터 로드
            print("Loading LoRA adapter...")
            _finetuned_model = PeftModel.from_pretrained(base_model, lora_path)
            print("Finetuned LoRA model loaded successfully.")
            
        except Exception as e:
            print(f"Warning: Could not load finetuned model: {e}")
            print("Falling back to base model.")
            # 파인튜닝된 모델 로드 실패시 기본 모델 사용
            return get_llm()
    
    if _finetuned_pipeline is None:
        try:
            print("Creating finetuned pipeline...")
            _finetuned_pipeline = pipeline("text-generation", model=_finetuned_model, tokenizer=_finetuned_tokenizer, max_new_tokens=1024, do_sample=False, temperature=0.1, return_full_text=False)
            print("Finetuned pipeline initialized.")
        except Exception as e:
            print(f"Error creating finetuned pipeline: {e}")
            print("Falling back to base model.")
            return get_llm()
    
    hf_pipeline = HuggingFacePipeline(pipeline=_finetuned_pipeline)
    return ChatHuggingFace(llm=hf_pipeline) 