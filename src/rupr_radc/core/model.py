import torch
from smolagents import TransformersModel

class BatchedQwenModel(TransformersModel):
    def __init__(self, config):
        super().__init__(model_id=config["model_path"], device_map="cuda:0",
                         torch_dtype=config["dtype"],
                         model_kwargs={"attn_implementation": config["attention"], "local_files_only": True},
                         **config["generation"])
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {
            "shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]}
        self.max_input_tokens = config["max_input_tokens"]
        self.structured_tokenizer_data = None

    def prepare_batch(self, conversations, allow_text_only=False):
        cleaned = []
        generation = None
        for conversation in conversations:
            kwargs = self._prepare_completion_kwargs(conversation, convert_images_to_image_urls=True)
            messages = kwargs.pop("messages")
            for message in messages:
                for item in message["content"]:
                    if item["type"] == "image_url":
                        item["type"] = "image"
                        item["image"] = item.pop("image_url")["url"]
            cleaned.append(messages)
            generation = kwargs
        inputs = self.processor.apply_chat_template(cleaned, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", padding=True)
        has_image = "pixel_values" in inputs and "image_grid_thw" in inputs
        requested_images = sum(item["type"] == "image" for messages in cleaned
                               for message in messages for item in message["content"])
        if requested_images and not has_image:
            raise RuntimeError("Image tensor missing: refusing to discard images")
        if not has_image and not allow_text_only:
            raise RuntimeError("Image tensor missing: refusing text-only evaluation")
        if has_image and len(inputs["image_grid_thw"]) != len(conversations):
            raise RuntimeError("Expected exactly one image per conversation")
        if not has_image and requested_images:
            raise RuntimeError("Text-only batch contains an image request")
        lengths = inputs["attention_mask"].sum(-1).tolist()
        if max(lengths) > self.max_input_tokens:
            raise RuntimeError(f"Input too long ({max(lengths)}); no silent truncation")
        return inputs.to(self.model.device), generation, lengths
