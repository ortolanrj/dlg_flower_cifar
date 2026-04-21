"""ServerApp — servidor malicioso que executa o ataque DLG.

O servidor usa uma estratégia personalizada (MaliciousFedAvg) que substitui
agreggate_train para:
  1. Intercepta todas as respostas do cliente.
  2. Escolha aleatoriamente um cliente como vítima.
  3. Recupere os gradientes da vítima a partir dos parâmetros retornados.
  4. Execute o ataque DLG para reconstruir a imagem privada da vítima.
  5. Prossegue com a agregação normal do FedAvg.
"""

import random
from typing import Iterable, Optional

import numpy as np
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from dlg_flower.task import LeNet, dlg_attack_from_gradients, weights_init


class MaliciousFedAvg(FedAvg):
    """FedAvg com um servidor malicioso que executa o ataque DLG em um cliente aleatório.

    O servidor é 'honesto-mas-curioso'.
    """

    def __init__(self, dlg_iterations: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.dlg_iterations = dlg_iterations
        self._server_arrays: Optional[ArrayRecord] = None

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Guarda os arrays enviados aos clientes para recuperar os gradientes depois."""
        self._server_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Intercepta replies, ataca um cliente aleatório (vítima), e depois agrega os dados normalmente."""
        replies_list = list(replies)

        if replies_list and self._server_arrays is not None:
            # ---- Passo 1: Escolhe um cliente aleatório ----
            victim_idx = random.randint(0, len(replies_list) - 1)
            victim_reply = replies_list[victim_idx]

            # Extrai metadados
            victim_metrics = victim_reply.content.get("metrics")
            victim_partition = int(victim_metrics["partition_id"]) if victim_metrics else "?"
            victim_label = int(victim_metrics["label"]) if victim_metrics else "?"

            print(f"\n{'#'*60}")
            print(f"  SERVER: Rodada {server_round}")
            print(f"  Recebido respostas de {len(replies_list)} clientes")
            print(f"  Selecionado cliente {victim_partition} randômicamente como vítima")
            print(f"  (Label verdadeira: {victim_label})")
            print(f"{'#'*60}")

            # ---- Passo 2: Recupera os gradientes da vítima ----
            # Enviado pelo cliente: params_new = params_old - gradient
            # Logo para termos o gradiente: gradient = params_old - params_new
            device = "cuda" if torch.cuda.is_available() else "cpu"

            server_state = self._server_arrays.to_torch_state_dict()
            victim_state = victim_reply.content["arrays"].to_torch_state_dict()

            recovered_gradients = []
            for key in server_state:
                g = (server_state[key] - victim_state[key]).to(device)
                recovered_gradients.append(g)

            # ---- Reconstrução do modelo ----
            net = LeNet().to(device)
            net.load_state_dict(
                {k: v.to(device) for k, v in server_state.items()}
            )
            net.eval()

            # ---- Ataque DLG ---- 
            dlg_attack_from_gradients(
                net=net,
                original_dy_dx=recovered_gradients,
                input_shape=(1, 3, 32, 32),
                num_classes=100,
                num_iterations=self.dlg_iterations,
                save_dir="dlg_results",
                client_id=victim_partition,
            )

        # ---- Passo 5: Agregação FedAvg ----
        return super().aggregate_train(server_round, iter(replies_list))


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Leitura do run_config
    num_rounds = int(context.run_config.get("num-server-rounds", 1))
    dlg_iterations = int(context.run_config.get("dlg-iterations", 300))

    # Inicialização do modelo
    torch.manual_seed(50)
    global_model = LeNet()
    global_model.apply(weights_init)

    # Cria ArrayRecord do state_dict do modelo
    arrays = ArrayRecord(global_model.state_dict())

    # Inicializa a Strategy
    strategy = MaliciousFedAvg(
        dlg_iterations=dlg_iterations,
        fraction_evaluate=1.0,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({}),
        num_rounds=num_rounds,
    )

    print("\n Término da execução do Aprendizado Federado.")
