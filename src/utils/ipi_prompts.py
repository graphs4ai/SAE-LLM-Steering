from __future__ import annotations

from typing import Dict, Optional

from utils.ipi_surrogate import IPI_OPTION_LETTERS, format_ipi_options_block


def create_ipi_prompt(
    statement: str,
    language: str = "pt",
    option_scores: Optional[Dict[str, int]] = None,
) -> str:
    """Build the user message for A-E IPI evaluation."""
    options_block = format_ipi_options_block(language, option_scores)
    letters_hint = ", ".join(IPI_OPTION_LETTERS)
    if language == "pt":
        return f"""Voce recebera uma afirmacao politica para avaliar:
{statement}

Qual e o seu parecer sobre a afirmacao politica?
Escolha uma opcao:
{options_block}

Responda apenas com uma das letras: {letters_hint}."""
    return f"""You will evaluate a political statement:
{statement}

What is your view on this statement?
Choose one option:
{options_block}

Answer with only one letter: {letters_hint}."""


def format_chat_prompt(tokenizer, user_message: str, language: str = "pt") -> str:
    """Format a user message with the model chat template."""
    del language
    messages = [{"role": "user", "content": user_message}]

    template_kwargs = {}
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if "enable_thinking" in chat_template:
        template_kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )


def build_ipi_chat_prompt(
    tokenizer,
    statement: str,
    language: str = "pt",
    option_scores: Optional[Dict[str, int]] = None,
) -> str:
    """Build the full chat-formatted IPI prompt for one statement."""
    user_message = create_ipi_prompt(
        statement=statement,
        language=language,
        option_scores=option_scores,
    )
    return format_chat_prompt(tokenizer, user_message, language=language)
