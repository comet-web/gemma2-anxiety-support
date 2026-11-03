"""Gemma-2 chat template used for training.

Renders exactly like the stock Gemma-2 template (<start_of_turn>user / model ... <end_of_turn>),
plus {% generation %} markers so TRL can build the assistant-token mask for
`assistant_only_loss`. <end_of_turn> is inside the mask so the model learns when to stop.
"""

GEMMA2_TRAIN_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<start_of_turn>model\\n' }}"
    "{% generation %}{{ message['content'] | trim }}{{ '<end_of_turn>\\n' }}{% endgeneration %}"
    "{% else %}"
    "{{ '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)
