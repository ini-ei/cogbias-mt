"""
OpenRouter経由でLLMにアクセスするクライアント

環境変数(.envファイル):
- OPENROUTER_API_KEY: OpenRouter APIキー
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()


class OpenRouterClient:
    """OpenRouter経由でLLMにアクセスするクライアント"""
    
    # 利用可能なモデル（2026-04-29更新 / 2026-05-12 オープン3機種追加）
    MODELS = {
        "gpt-5": "openai/gpt-5.4",                                    # 既存被験 + 評価者
        "gpt-5.5": "openai/gpt-5.5",                                  # 追加被験（フィードバック対応）
        "claude-4-sonnet": "anthropic/claude-sonnet-4.6",             # 既存被験 + 評価者
        "claude-opus-4.7": "anthropic/claude-opus-4.7",               # 追加被験（フィードバック対応）
        "gemini-3.1-pro": "google/gemini-3.1-pro-preview-20260219",   # 既存被験 + #12 で追加クロス評価者
        "llama-4-maverick": "meta-llama/llama-4-maverick",            # 既存被験
        # --- Phase 7 オープンモデル (DGX A100 で vLLM serve、OPENAI_BASE_URL でローカル切替) ---
        # 2026-05-13 更新: Qwen3.5 系・Gemma-4 系の最新世代
        "qwen3.5-27b":    "Qwen/Qwen3.5-27B",
        "gemma-4-31b":    "google/gemma-4-31B-it",
        "qwen3.5-122b":   "Qwen/Qwen3.5-122B-A10B",
        # --- Fallback (forward-compat が動かなかった場合の保険) ---
        "qwen2.5-72b":    "Qwen/Qwen2.5-72B-Instruct",
        "gemma-2-27b":    "google/gemma-2-27b-it",
        "llama-3.3-70b":  "meta-llama/Llama-3.3-70B-Instruct",
    }
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-5",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        初期化
        
        Args:
            api_key: OpenRouter APIキー (Noneの場合は.envから取得)
            model: 使用するモデル名 (簡易名またはフルパス)
            base_url: OpenRouter APIのベースURL
        """
        # .envファイルから読み込み
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter APIキーが設定されていません。\n"
                ".envファイルに OPENROUTER_API_KEY を設定するか、api_keyパラメータを指定してください。"
            )
        
        # モデル名の解決
        self.model = self.MODELS.get(model, model)
        
        # OpenAI互換クライアントの作成
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
    
    def chat_completion(self, messages: list, **kwargs) -> str:
        """
        チャット補完を実行
        
        Args:
            messages: メッセージリスト
            **kwargs: その他のパラメータ (temperature, max_tokens等)
            
        Returns:
            生成されたテキスト
        """
        # デフォルトパラメータ
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        # オプションパラメータを追加
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        
        response = self.client.chat.completions.create(**params)
        
        return response.choices[0].message.content
    
    def get_client(self):
        """
        OpenAI互換クライアントを取得
        
        Returns:
            OpenAIクライアント
        """
        return self.client


def create_openrouter_client(model: str = "gpt-5"):
    """
    OpenAI 互換クライアントを作成。

    動作モード:
      - LOCAL_MODEL が設定されている場合: transformers で local model を load し、
        OpenAI duck-type 互換 facade を返す (vllm 不要、HTTP 不要)。
        例: LOCAL_MODEL="Qwen/Qwen3.5-27B"
      - OPENAI_BASE_URL が設定されている場合: ローカル vLLM 等の OpenAI 互換エンドポイント
        (例: http://localhost:8000/v1)。
      - いずれも未設定: OpenRouter (https://openrouter.ai/api/v1)。

    Args:
        model: 使用するモデル名 (情報目的)

    Returns:
        OpenAI 互換クライアント
    """
    local_model = os.getenv("LOCAL_MODEL")
    if local_model:
        # transformers で local model を load (singleton で reuse)
        from utils.local_model_client import LocalClient
        return LocalClient.get_or_create(local_model)

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")

    if base_url:
        # ローカル vLLM 等。api_key 無くても OK (vLLM はデフォルトで認証無し)。
        client = OpenAI(
            api_key=api_key or "dummy",
            base_url=base_url,
            timeout=300.0,  # 大モデル + 長文出力で時間がかかる場合がある
        )
        return client

    # OpenRouter (default)
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY が設定されていません。\n"
            "OpenRouter 利用時は OPENROUTER_API_KEY を、ローカル vLLM 利用時は OPENAI_BASE_URL を設定してください。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=120.0,
    )

    return client


def get_model_name(model: str = "gpt-5") -> str:
    """
    モデル名を解決
    
    Args:
        model: 簡易モデル名またはフルパス
        
    Returns:
        OpenRouter用のフルモデル名
    """
    return OpenRouterClient.MODELS.get(model, model)


def main():
    """使用例"""
    # クライアントの作成
    client = OpenRouterClient(model="gpt-5")
    
    # チャット補完
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "あなたは親切なアシスタントです。"},
            {"role": "user", "content": "こんにちは!"}
        ],
        temperature=0.7
    )
    
    print(f"モデル: {client.model}")
    print(f"応答: {response}")


if __name__ == "__main__":
    main()

