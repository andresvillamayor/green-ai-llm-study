"""
prompt_builder.py
Proyecto GREEN-IA — Maestria Ciencia de Datos
Universidad Comunero — Paraguay
Autor: Andres Villamayor

Construye el string de prompt listo para pasar a llama-cpp-python (completado raw).
Cada modelo instruct/chat fue entrenado con un template especifico; usarlo
incorrectamente degrada la calidad de respuesta y puede aumentar el consumo
energetico, ya que el modelo genera tokens adicionales para compensar una
estructura de entrada malformada.

Templates soportados:
  qwen*        : ChatML (<|im_start|>/<|im_end|>) — Qwen2, Qwen2.5 instruct
  llama-2*chat : Llama-2 Chat ([INST]/<<SYS>>) — solo variante -chat/-instruct
  (resto)      : fallback de texto plano con separadores [Turn N]

Uso basico:

    from src.prompt_builder import build_prompt

    # Turno 1
    messages = [{"role": "user", "content": question["turn1"]}]
    prompt = build_prompt(model_name, messages)

    # Turno 2 (multi-turno MT-Bench)
    messages = [
        {"role": "user",      "content": question["turn1"]},
        {"role": "assistant", "content": t1_response},
        {"role": "user",      "content": question["turn2"]},
    ]
    prompt = build_prompt(model_name, messages)

Campos de messages:
    role    : "system" | "user" | "assistant"
    content : texto verbatim (P5, P6 — no modificar prompts MT-Bench)
"""

from __future__ import annotations

__all__ = ["build_prompt"]

_QWEN_DEFAULT_SYSTEM = "You are a helpful assistant."


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def build_prompt(model_name: str, messages: list[dict]) -> str:
    """
    Build a raw prompt string from a message list for the given model.

    Args:
        model_name: model identifier as used in config.yaml, e.g.
                    "qwen2.5-7b", "llama-2-7b", "llama-2-7b-chat".
        messages:   list of {"role": str, "content": str} dicts.
                    Roles: "system", "user", "assistant".
                    The last message must be from "user".

    Returns:
        Formatted prompt string ready for llama-cpp-python text completion.

    Raises:
        ValueError: if messages is empty or the last message is not from "user".

    WARNING: An incorrect chat template reduces response quality and increases
    energy consumption — a malformed prompt causes the model to generate extra
    tokens attempting to recover the expected structure.
    """
    if not messages:
        raise ValueError("build_prompt: messages list is empty")
    if messages[-1].get("role") != "user":
        raise ValueError(
            "build_prompt: the last message must have role='user'; "
            f"got role='{messages[-1].get('role')}'"
        )

    name = model_name.lower()

    if "qwen" in name:
        return _build_qwen_prompt(messages)

    if "llama-2" in name and any(tag in name for tag in ("chat", "instruct")):
        return _build_llama2_chat_prompt(messages)

    return _build_generic_prompt(messages)


# ─── Qwen2 / Qwen2.5 Instruct — ChatML ────────────────────────────────────────

def _build_qwen_prompt(messages: list[dict]) -> str:
    """
    ChatML format used by Qwen2 and Qwen2.5 Instruct models.

    Structure:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {content}<|im_end|>
        <|im_start|>assistant
        {content}<|im_end|>
        ...
        <|im_start|>assistant
                            <- model completes from here

    A default system message is injected if none is provided in messages,
    because Qwen2.5-Instruct was fine-tuned expecting the <|im_start|>system
    token; omitting it produces lower-quality outputs.

    Reference: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct (chat_template)
    """
    parts: list[str] = []

    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        parts.append(
            f"<|im_start|>system\n{_QWEN_DEFAULT_SYSTEM}<|im_end|>\n"
        )

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ─── Llama-2 Chat ──────────────────────────────────────────────────────────────

def _build_llama2_chat_prompt(messages: list[dict]) -> str:
    """
    Official Llama-2 Chat template from Meta AI.

    Single-turn (no system):
        <s>[INST] {user} [/INST]

    Single-turn (with system):
        <s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user} [/INST]

    Multi-turn:
        <s>[INST] {user_1} [/INST] {assistant_1} </s>
        <s>[INST] {user_2} [/INST]

    The system message, if present, is embedded inside the first [INST] block.
    Applicable only to the -chat or -instruct variants of Llama-2;
    the base model does not use this template.

    Reference: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """
    system_content: str | None = None
    turns: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_content = msg.get("content", "")
        else:
            turns.append(msg)

    parts: list[str] = []
    user_buf: str | None = None

    for msg in turns:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            if user_buf is not None:
                # Previous user turn had no assistant reply — flush it open-ended
                parts.append(f"<s>[INST] {user_buf} [/INST]")
            user_buf = content

        elif role == "assistant":
            if user_buf is None:
                continue  # malformed: assistant without prior user turn
            if not parts and system_content is not None:
                # Embed system into the first [INST] block
                user_block = f"<<SYS>>\n{system_content}\n<</SYS>>\n\n{user_buf}"
            else:
                user_block = user_buf
            parts.append(f"<s>[INST] {user_block} [/INST] {content} </s>")
            user_buf = None

    # Flush final user turn (model completes from here)
    if user_buf is not None:
        if not parts and system_content is not None:
            user_block = f"<<SYS>>\n{system_content}\n<</SYS>>\n\n{user_buf}"
        else:
            user_block = user_buf
        parts.append(f"<s>[INST] {user_block} [/INST]")

    return "".join(parts)


# ─── Generic fallback ─────────────────────────────────────────────────────────

def _build_generic_prompt(messages: list[dict]) -> str:
    """
    Plain-text fallback for models without a specific chat template.

    Uses labeled sections that are readable and unambiguous without relying
    on any special tokens. Suitable for base models (e.g., llama-2-7b base)
    and for smoke-testing unknown model names.

    Multi-turn MT-Bench example output:
        [Turn 1 Question]
        {turn1_question}

        [Turn 1 Answer]
        {turn1_response}

        [Turn 2 Question]
        {turn2_question}
    """
    system_parts: list[str] = []
    turn_parts:   list[str] = []
    user_count = 0

    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_count += 1
            turn_parts.append(f"[Turn {user_count} Question]\n{content}")
        elif role == "assistant":
            turn_parts.append(f"[Turn {user_count} Answer]\n{content}")

    sections: list[str] = []
    if system_parts:
        sections.append("\n\n".join(system_parts))
    sections.extend(turn_parts)

    return "\n\n".join(sections)
