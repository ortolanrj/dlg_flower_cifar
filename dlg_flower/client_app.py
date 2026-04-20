"""
Cliente 0 (partition_id == 0):
    A "vítima" — recebe o modelo e o atacante intercepta seus
gradientes para executar o ataque de reconstrução DLG.

Cliente 1 (partition_id == 1):
    Cliente benigno que treina normalmente.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from dlg_flower.task import (
    LeNet,
    dlg_attack,
    get_parameters,
    load_cifar100,
    set_parameters,
    tp,
)


class DLGClient(NumPyClient):
    def __init__(self, partition_id: int, run_config: dict):
        super().__init__()
        self.partition_id = partition_id
        self.run_config = run_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LeNet().to(self.device)
        self.dst = load_cifar100()

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        if self.partition_id == 0:
            # ---- Cliente Vítima: o atacante intercepta os gradientes ----
            img_index = int(self.run_config.get("target-index", 25))
            dlg_iters = int(self.run_config.get("dlg-iterations", 300))

            self.net.eval()

            dlg_attack(
                net=self.net,
                dst=self.dst,
                img_index=img_index,
                num_iterations=dlg_iters,
                save_dir="dlg_results",
            )

            return get_parameters(self.net), 1, {"partition_id": 0}
        else:
            # ---- Cliente benígno: treinamento normal ----
            self.net.train()
            indices = list(range(128))
            images = []
            labels = []
            for i in indices:
                img = tp(self.dst[i][0]).to(self.device)
                lab = self.dst[i][1]
                images.append(img)
                labels.append(lab)
            images = torch.stack(images)
            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
            for start in range(0, len(images), 32):
                batch_img = images[start:start+32]
                batch_lab = labels[start:start+32]
                optimizer.zero_grad()
                loss = F.cross_entropy(self.net(batch_img), batch_lab)
                loss.backward()
                optimizer.step()

            return get_parameters(self.net), len(images), {"partition_id": 1}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        self.net.eval()
        correct, total, total_loss = 0, 0, 0.0

        with torch.no_grad():
            for i in range(256):
                img = tp(self.dst[i][0]).unsqueeze(0).to(self.device)
                lab = torch.tensor([self.dst[i][1]], dtype=torch.long, device=self.device)
                out = self.net(img)
                total_loss += F.cross_entropy(out, lab).item()
                correct += (out.argmax(1) == lab).sum().item()
                total += 1

        return total_loss / total, total, {"accuracy": correct / total}


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    return DLGClient(
        partition_id=partition_id,
        run_config=context.run_config,
    ).to_client()


app = ClientApp(client_fn=client_fn)
