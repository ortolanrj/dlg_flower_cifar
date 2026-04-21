"""ClientApp — todos os clientes são participantes honestos

Cada cliente:
  1. Recebe o modelo global via Message contendo um ArrayRecord.
  2. Pega uma imagem privada baseada no partition_id.
  3. Calcula o gradiente nesta imagem.
  4. Retorna os gradientes serializados como parâmetros de update em um ArrayRecord,
     e metadados (partition_id, label) em um MetricRecord.
"""

import torch
import torch.nn.functional as F

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from dlg_flower.task import (
    LeNet,
    cross_entropy_for_onehot,
    label_to_onehot,
    load_cifar100,
    tp,
)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega modelo do ArrayRecord do servidor
    net = LeNet().to(device)
    server_arrays = msg.content["arrays"]
    net.load_state_dict(server_arrays.to_torch_state_dict())
    net.eval()

    # Determina qual imagem este cliente utiliza (baseado no partition_id)
    partition_id = context.node_config["partition-id"]
    img_index = partition_id

    # Carrega dataset CIFAR-100
    dst = load_cifar100()
    criterion = cross_entropy_for_onehot

    gt_data = tp(dst[img_index][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label, num_classes=100)

    # Calcula o gradiente desta imagem
    out = net(gt_data)
    y = criterion(out, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    gradients = [g.detach() for g in dy_dx]

    # Serializa gradientes como parâmetros atualizados: params_new = params_old - gradient
    # O servidor recupera: gradient = params_old - params_new
    original_state = server_arrays.to_torch_state_dict()
    updated_state = {}
    grad_list = list(gradients)
    for i, (key, param) in enumerate(original_state.items()):
        updated_state[key] = param - grad_list[i]

    # Constróis reply Message
    updated_arrays = ArrayRecord(updated_state)
    metrics = MetricRecord({
        "partition_id": partition_id,
        "img_index": img_index,
        "label": int(gt_label.item()),
        "num-examples": 1,
    })
    content = RecordDict({"arrays": updated_arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LeNet().to(device)
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    net.eval()

    dst = load_cifar100()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for i in range(min(256, len(dst))):
            img = tp(dst[i][0]).unsqueeze(0).to(device)
            lab = torch.tensor([dst[i][1]], dtype=torch.long, device=device)
            out = net(img)
            total_loss += F.cross_entropy(out, lab).item()
            correct += (out.argmax(1) == lab).sum().item()
            total += 1

    metrics = MetricRecord({
        "accuracy": correct / total,
        "loss": total_loss / total,
        "num-examples": total,
    })
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
