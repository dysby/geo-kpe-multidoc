from transformers import LongformerModel

model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
import torch

torch_device = "cpu"
input_ids = torch.tensor(
    [[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device
)  # long input
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
global_attention_mask = torch.zeros(
    input_ids.shape, dtype=torch.long, device=input_ids.device
)
global_attention_mask[
    :, [1, 4, 21]
] = 1  # Set global attention on a few random positions

output = model(
    input_ids,
    attention_mask=attention_mask,
    global_attention_mask=global_attention_mask,
)[0]

expected_output_sum = torch.tensor(74585.8594, device=torch_device)
expected_output_mean = torch.tensor(0.0243, device=torch_device)

print(torch.allclose(output.sum(), expected_output_sum, atol=1e-4))

print(torch.allclose(output.mean(), expected_output_mean, atol=1e-4))
