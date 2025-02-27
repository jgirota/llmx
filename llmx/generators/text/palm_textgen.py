from dataclasses import asdict
import os
import logging
from typing import Dict, Union, List
import requests
import json  # Importe a biblioteca json
from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import (
    cache_request,
    gcp_request,  # Mantenha, mas vamos modificar
    get_models_maxtoken_dict,
    num_tokens_from_messages,
    get_gcp_credentials,
)

logger = logging.getLogger("llmx")


class PalmTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("PALM_API_KEY", None),
        palm_key_file: str = os.environ.get("PALM_SERVICE_ACCOUNT_KEY_FILE", None),
        project_id: str = os.environ.get("PALM_PROJECT_ID", None),
        project_location=os.environ.get("PALM_PROJECT_LOCATION", "us-central1"),
        provider: str = "palm",  # Mantém o provider como "palm" para compatibilidade
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)

        if api_key is None and palm_key_file is None:
            raise ValueError(
                "PALM_API_KEY or PALM_SERVICE_ACCOUNT_KEY_FILE must be set."
            )
        if api_key:
            self.api_key = api_key
            self.credentials = None
            self.project_id = None
            self.project_location = None
        else:
            self.project_id = project_id
            self.project_location = project_location
            self.api_key = None
            self.credentials = get_gcp_credentials(palm_key_file) if palm_key_file else None

        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "gemini-1.5-pro-002"  # Modelo padrão Gemini (alterado)


    def format_messages(self, messages: List[Dict]) -> str:
        """Converts a list of messages into a single string for Gemini."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt += f"{content}\n"  # System messages as context
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        return prompt


    def generate(
        self,
        messages: Union[list[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name

        # Format messages for Gemini
        prompt = self.format_messages(messages)
        self.model_name = model

        max_tokens = self.model_max_token_dict[model] if model in self.model_max_token_dict else 2048
        temperature = config.temperature
        n = config.n  # Number of candidates

        # Limitar o valor de topK
        top_k = min(config.top_k, 40)  # Garante que top_k não seja maior que 40

        # Use Vertex AI Gemini endpoint
        api_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.project_location}/publishers/google/models/{model}:generateContent"

        payload = {
            "contents": [{
                "role": "USER",
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topP": config.top_p,
                "topK": top_k,  # Use o valor limitado de top_k
                "maxOutputTokens": config.max_tokens or max_tokens,
            },
            "safetySettings": [  # Configurações de segurança (opcional, mas recomendado)
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }

        cache_key_params = {**payload, "model": model, "api_url": api_url}

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        # Use the existing gcp_request function, but adapt the parameters
        palm_response = gcp_request(
            url=api_url, body=payload, method="POST", credentials=self.credentials
        )

        candidates = []
        for candidate in palm_response.get("candidates", []):
            # Extrai o texto corretamente
            content = candidate["content"]["parts"][0]["text"]
            candidates.append(Message(role="assistant", content=content))

        response_text = candidates

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],
            config=asdict(config), #Usando asdict para serializar corretamente
            usage={
                "total_tokens": num_tokens_from_messages(
                    [{"role": m.role, "content": m.content} for m in response_text], model=self.model_name
                )
            },
            response=palm_response,
        )

        cache_request(
            cache=self.cache, params=(cache_key_params), values=asdict(response)
        )
        return response


    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
