"""Restore the Transformers 4 chat-template return contract for frozen callers."""
TOKENIZER_COMPATIBILITY = {'apply_chat_template_default_return_dict': False}


class ListChatTokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault('return_dict', False)
        return self._tokenizer.apply_chat_template(*args, **kwargs)
