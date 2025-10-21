import random
import heapq
import torch
import numpy as np
import time
import os
import sys
import json

class NextPointNet(torch.nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=1,
                 num_hidden_layers=3, activation='leakyrelu',
                 dropout_rate=0.2, batch_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = max(1, num_hidden_layers)
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        if activation.lower() == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            self.activation = torch.nn.LeakyReLU()
        elif activation.lower() == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.ReLU()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(torch.nn.Dropout(dropout_rate))
        for _ in range(self.num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class AiML:
    def __init__(self, points, graph, parent_class=None):
        self.parent_class = parent_class
        self.points = points
        self.graph = graph
        self.epochs = 5000
        self.training_data = 2000
        self.nn_config = {
            'input_dim': 6,
            'hidden_dim': 64,
            'output_dim': 1,
            'num_hidden_layers': 2,
            'activation': 'leakyrelu',
            'dropout_rate': 0.2,
            'batch_norm': False,
            'learning_rate': 0.01,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'epochs': self.epochs,
            'optimizer': 'adamw',
            'scheduler_step': 10,
            'scheduler_gamma': 0.9,
            'gradient_clip': 1.0
        }
        self._initialize_class_mappings()

    def _initialize_class_mappings(self):
        if not hasattr(self, 'points') or not self.points:
            print("Erreur : Aucun point défini pour créer les mappages ID/Classe")
            return False
        all_point_ids = sorted(self.points.keys())
        self.unique_ids = all_point_ids
        self.id_to_class = {pid: idx for idx, pid in enumerate(self.unique_ids)}
        self.class_to_id = {idx: pid for idx, pid in enumerate(self.unique_ids)}
        self.num_classes = len(self.unique_ids)
        if 'output_dim' in self.nn_config:
            self.nn_config['output_dim'] = self.num_classes
        print(f"[AI] Mappages ID/Classe initialisés pour {self.num_classes} points")
        return True

    def _calculate_distance(self, p1, p2):
        return sum((a - b)**2 for a, b in zip(p1, p2))**0.5

    def astar(self, start, goal):
        point_dict = {pid: list(data['pos']) for pid, data in self.points.items()}
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph}
        f_score[start] = self._calculate_distance(point_dict[start], point_dict[goal])

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            for neighbor in self.graph[current]:
                tentative_g_score = g_score[current] + self._calculate_distance(point_dict[current], point_dict[neighbor])
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._calculate_distance(point_dict[neighbor], point_dict[goal])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _generate_training_data(self, num_samples=1000):
        num_samples = self.training_data
        print("[AI] Début de la génération des données d'entraînement")
        if not hasattr(self, 'id_to_class'):
            self._initialize_class_mappings()
        self.X_data = []
        self.y_data = []
        valid_ids = list(self.points.keys())
        if not valid_ids:
            print("[ERREUR] Aucun point valide trouvé dans le graphe")
            return False

        for _ in range(num_samples):
            ai_id = random.choice(valid_ids)
            player_id = random.choice(valid_ids)

            # S'assurer que les points ne sont pas les mêmes
            if ai_id == player_id:
                continue

            path = self.astar(ai_id, player_id)

            # Si un chemin est trouvé et qu'il a au moins deux points
            if path and len(path) >= 2:
                ai_pos = self.points[ai_id]['pos']
                player_pos = self.points[player_id]['pos']
                input_vec = list(ai_pos) + list(player_pos)
                self.X_data.append(input_vec)
                next_id = path[1]  # Le prochain point après le départ
                self.y_data.append(self.id_to_class[next_id])

        print(f"Génération terminée. {len(self.X_data)} échantillons valides générés.")
        return True

    def _prepare_training(self):
        print("[AI] Préparation des outils d'entraînement...")
        if not self._verify_data_integrity():
            print("[ERREUR] Problèmes avec les données d'entraînement")
            return False
        try:
            X_tensor = torch.tensor(self.X_data, dtype=torch.float32)
            y_tensor = torch.tensor(self.y_data, dtype=torch.long)
            self._train_state = {
                'X_data': X_tensor,
                'y_data': y_tensor,
                'data_size': len(X_tensor)
            }
        except Exception as e:
            print(f"Erreur lors de la conversion des données en tenseurs: {e}")
            return False
        try:
            self.next_point_to_aitarg_model = NextPointNet(
                input_dim=self.nn_config['input_dim'],
                hidden_dim=self.nn_config['hidden_dim'],
                output_dim=self.nn_config['output_dim'],
                num_hidden_layers=self.nn_config['num_hidden_layers'],
                activation=self.nn_config['activation'],
                dropout_rate=self.nn_config['dropout_rate'],
                batch_norm=self.nn_config['batch_norm']
            )
            self._train_state['model_created'] = True
        except Exception as e:
            print(f"Erreur lors de la création du modèle: {e}")
            return False
        try:
            params = self.next_point_to_aitarg_model.parameters()
            lr = self.nn_config['learning_rate']
            wd = self.nn_config['weight_decay']
            if self.nn_config['optimizer'].lower() == 'adam':
                optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
            elif self.nn_config['optimizer'].lower() == 'adamw':
                optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            elif self.nn_config['optimizer'].lower() == 'sgd':
                optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
            else:
                optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            self._train_state['optimizer'] = optimizer
        except Exception as e:
            print(f"Erreur lors de la configuration de l'optimiseur: {e}")
            return False
        try:
            if hasattr(self, 'y_data') and self.y_data:
                y_array = np.array(self.y_data)
                unique_classes, counts = np.unique(y_array, return_counts=True)
                if len(unique_classes) > 1:
                    # Création des poids pour chaque classe connue dans le dataset
                    class_weights = np.zeros(self.num_classes, dtype=np.float32)
                    for class_idx, count in zip(unique_classes, counts):
                        class_weights[class_idx] = 1.0 / count
                    class_weights = class_weights / np.sum(class_weights)
                    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

                    if len(class_weights_tensor) == self.num_classes:
                        self._train_state['criterion'] = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
                        print(f"[AI] Fonction de perte avec poids personnalisés sur {self.num_classes} classes.")
                    else:
                        print(f"[AI] ⚠️ Taille incohérente des poids ({len(class_weights_tensor)} != {self.num_classes}). Utilisation sans poids.")
                        self._train_state['criterion'] = torch.nn.CrossEntropyLoss()
                else:
                    self._train_state['criterion'] = torch.nn.CrossEntropyLoss()
                    print(f"[AI] Fonction de perte sans pondération.")

        except Exception as e:
            print(f"Erreur lors de la configuration de la fonction de perte: {e}")
            self._train_state['criterion'] = torch.nn.CrossEntropyLoss()
        try:
            self._train_state['scheduler'] = torch.optim.lr_scheduler.StepLR(
                self._train_state['optimizer'],
                step_size=self.nn_config['scheduler_step'],
                gamma=self.nn_config['scheduler_gamma']
            )
        except Exception as e:
            print(f"Avertissement: Impossible de configurer le scheduler: {e}")
            self._train_state['scheduler'] = None
        print("[AI] Préparation terminée. Tout est prêt pour l'entraînement.")
        return True

    def _verify_data_integrity(self):
        if not hasattr(self, 'X_data') or not hasattr(self, 'y_data'):
            return False
        print("[AI] Vérification de l'intégrité des données...")
        valid = True
        max_class_index = self.num_classes - 1
        y_data_valid = all(0 <= y <= max_class_index for y in self.y_data)
        if not y_data_valid:
            print("Erreur: Des indices de classe invalides dans y_data")
            valid = False
        missing_classes = [y for y in self.y_data if y not in self.class_to_id]
        if missing_classes:
            print(f"Erreur: {len(missing_classes)} indices de classe invalides (max {max_class_index})")
            valid = False
        if valid:
            print("[AI] Vérification de l'intégrité des données réussie")
        else:
            print("[ERREUR] Problèmes d'intégrité des données détectés")
        return valid

    def _train_epoch(self):
        if not self._train_state.get('model_created', False):
            print("[ERREUR] Le modèle n'est pas prêt pour l'entraînement")
            return None
        model = self.next_point_to_aitarg_model
        criterion = self._train_state['criterion']
        optimizer = self._train_state['optimizer']
        scheduler = self._train_state.get('scheduler', None)
        batch_size = self._train_state.get('batch_size', 64)
        data_size = self._train_state['data_size']
        X_data = self._train_state['X_data']
        y_data = self._train_state['y_data']
        model.train()
        total_loss = 0.0
        num_batches = 0
        indices = torch.randperm(data_size)
        X_shuffled = X_data[indices]
        y_shuffled = y_data[indices]
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            if 'gradient_clip' in self.nn_config and self.nn_config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.nn_config['gradient_clip'])
            optimizer.step()
            total_loss += loss.item() * (end_idx - start_idx)
            num_batches += 1
            if start_idx % (batch_size * 10) == 0:
                current_loss = total_loss / (end_idx) if end_idx > 0 else 0
                progress = (end_idx / data_size) * 100
                print(f"Progrès: {progress:.1f}% - Perte actuelle: {current_loss:.4f}")
        epoch_loss = total_loss / data_size if data_size > 0 else 0
        if scheduler:
            scheduler.step()
        return epoch_loss

    def _train_model(self, epochs):
        print(f"\n[AI] Début de l'entraînement du modèle pour {epochs} époques")
        best_loss = float('inf')
        patience = 500
        epochs_without_improvement = 0
        history = {'epoch_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = self._train_epoch()
            if epoch_loss is None:
                print("Erreur lors de l'entraînement de l'époque")
                break
            history['epoch_loss'].append(epoch_loss)
            epoch_time = time.time() - epoch_start_time
            print(f"Époque {epoch+1}/{epochs} - Perte: {epoch_loss:.4f} - Temps: {epoch_time:.2f}s")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                self.save_model()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Arrêt anticipé après {epochs_without_improvement} époques sans amélioration")
                    break
        print("\nEntraînement terminé")
        return history

    def train(self):
        epochs = self.nn_config.get('epochs', 5000)  # Valeur par défaut si 'epochs' n'est pas défini dans nn_config

        # Initialiser _train_state avec toutes les clés nécessaires
        self._train_state = {
            'epoch': 5000,
            'epochs': epochs,
            'data_generated': False,
            'training_phase': 'init',
            'batch_size': self.nn_config.get('batch_size', 64)
        }

        if not self._train_state.get('data_generated', False):
            print("\n[Phase 1/3] Génération des données d'entraînement")
            success = self._generate_training_data()
            if not success:
                print("[ERREUR] Échec de la génération des données")
                return
            print("\n[Phase 2/3] Préparation de l'entraînement")
            if not self._prepare_training():
                print("[ERREUR] Échec de la préparation de l'entraînement")
                return
            self._train_state['data_generated'] = True
            self._train_state['training_phase'] = 'ready_to_train'

        if self._train_state.get('training_phase') == 'ready_to_train':
            print("\n[Phase 3/3] Entraînement du modèle")
            epochs = self._train_state.get('epochs', self.nn_config.get('epochs', 5000))
            history = self._train_model(epochs)
            self._train_state['training_phase'] = 'complete'
            self._train_state['training_history'] = history
            self.save_model()



    def save_model(self, filepath="next_point_model.pth"):
        if not hasattr(self, 'next_point_to_aitarg_model'):
            print("[ERREUR] Aucun modèle à sauvegarder")
            return False
        try:
            checkpoint = {
                'model_state_dict': self.next_point_to_aitarg_model.state_dict(),
                'id_to_class': self.id_to_class,
                'class_to_id': self.class_to_id,
                'nn_config': self.nn_config,
                'optimizer_state_dict': self._train_state['optimizer'].state_dict() if hasattr(self, '_train_state') and 'optimizer' in self._train_state else None,
                'epoch': self._train_state.get('epoch', 0),
                'loss_history': self._train_state.get('epoch_stats', []),
                'training_history': self._train_state.get('training_history', {})
            }
            torch.save(checkpoint, filepath)
            print(f"[AI] Modèle sauvegardé dans {filepath}")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle: {e}")
            return False

    def load_model(self, filepath="next_point_model.pth"):
        if not os.path.exists(filepath):
            print(f"Fichier modèle introuvable: {filepath}")
            return False
        try:
            checkpoint = torch.load(filepath)

            # Vérifie la compatibilité entre les classes sauvegardées et actuelles
            if 'id_to_class' in checkpoint:
                saved_num_classes = len(checkpoint['id_to_class'])
                current_num_classes = len(self.points)
                if saved_num_classes != current_num_classes:
                    print(f"[ML] Incompatibilité détectée : modèle a {saved_num_classes} classes, mais le graphe a {current_num_classes}. Ignoré.")
                    return False

                self.id_to_class = checkpoint['id_to_class']
                self.class_to_id = checkpoint.get('class_to_id', {v: k for k, v in self.id_to_class.items()})
                self.unique_ids = list(sorted(self.id_to_class.keys()))
                self.num_classes = len(self.unique_ids)
                print(f"[ML] Mappages ID/Classe chargés avec {self.num_classes} classes")

            self.next_point_to_aitarg_model = NextPointNet(
                input_dim=self.nn_config.get('input_dim', 6),
                hidden_dim=self.nn_config.get('hidden_dim', 64),
                output_dim=self.nn_config.get('output_dim', self.num_classes),
                num_hidden_layers=self.nn_config.get('num_hidden_layers', 2),
                activation=self.nn_config.get('activation', 'leakyrelu'),
                dropout_rate=self.nn_config.get('dropout_rate', 0.2),
                batch_norm=self.nn_config.get('batch_norm', False)
            )
            self.next_point_to_aitarg_model.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint and hasattr(self, '_train_state'):
                if 'optimizer' in self._train_state:
                    self._train_state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

            if 'training_history' in checkpoint:
                if not hasattr(self, '_train_state'):
                    self._train_state = {}
                self._train_state['training_history'] = checkpoint['training_history']

            print(f"[ML] Modèle et configuration chargés depuis {filepath}")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False



    def predict(self, ai_pos, targ_pos):
        if not hasattr(self, 'next_point_to_aitarg_model'):
            print("[ML] No model available.")
            return None
        input_tensor = torch.tensor([[ai_pos.x, ai_pos.y, ai_pos.z, targ_pos.x, targ_pos.y, targ_pos.z]], dtype=torch.float32)
        with torch.no_grad():
            logits = self.next_point_to_aitarg_model(input_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()
        predicted_id = self.class_to_id.get(predicted_class, None)
        print(f"[ML] Predicted class: {predicted_class}, point ID: {predicted_id}")
        return predicted_id if predicted_id in self.points else None

    def ML_seek(self, task, ai_node, aitarg):
        if not aitarg:
            return task.again
        ai_pos = ai_node.getPos()
        targ_pos = aitarg.getPos()
        self.predicted_next_id = self.predict(ai_pos, targ_pos)
        if self.predicted_next_id in self.points:
            print(f"[AI] ML predicted next point: {self.predicted_next_id}")
            return self.predicted_next_id
        else:
            print("[AI] ML prediction invalid, no move.")
            return None
