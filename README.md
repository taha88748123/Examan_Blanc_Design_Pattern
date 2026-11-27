# Examen Blanc Design Pattern — Réseau de neurones artificiels

Implémentation Java (Maven) d'un réseau de neurones entièrement connecté, conçue autour des
design patterns demandés dans l'énoncé de l'examen **Design Pattern (GLSID — ENSET Mohammedia)**.

## Design patterns mis en œuvre

| Exigence | Pattern | Rôle dans le projet |
|---|---|---|
| a. Famille de fonctions d'activation interchangeables | **Strategy** | `ActivationFunction` + `Sigmoid`, `ReLU`, `Tanh`, `Identity`, `Softmax` |
| b. Adaptation d'une ancienne fonction d'activation | **Adapter** | `LegacyActivationAdapter` adapte `LegacySquashingFunction` |
| c. États Construction / Training / Trained | **State** | `NeuralNetworkState` + `ConstructionState`, `TrainingState`, `TrainedState` |
| d. Notification des composants dépendants | **Observer** | `NetworkObserver` + `Dashboard` |
| e. Comportement dynamique des couches | **Decorator** | `LayerDecorator`, `DropoutDecorator`, `BatchNormalizationDecorator` |
| f. Journalisation de toutes les méthodes | **Aspect (AOP)** | `LoggingAspect` (AspectJ) |
| g. Sécurisation de l'application | **Aspect (AOP)** | `SecurityAspect` + `@Secured` |
| Améliorations | **Factory, Builder, Singleton** | `LayerFactory`, `ActivationFactory`, `NeuralNetworkBuilder`, `NetworkConfig` |

## Structure du projet

```
src/main/java/ma/enset/nn/
├── Main.java                  # Démonstration exécutable
├── core/                      # NeuralNetwork, Layer, Neuron, Connection
├── activation/                # Strategy + Adapter
├── state/                     # State
├── observer/                  # Observer
├── decorator/                 # Decorator
├── factory/                   # Factory
├── builder/                   # Builder
├── config/                    # Singleton
└── aspect/                    # LoggingAspect, SecurityAspect, @Secured
```

## Compilation, tests et exécution

```bash
mvn clean test          # compile, tisse les aspects et exécute les tests JUnit 5
mvn exec:java           # lance la démonstration (Main)
```

Le rapport complet (conception, code, tests, captures d'écran) est disponible dans
[`docs/rapport.pdf`](docs/rapport.pdf) et l'énoncé dans [`docs/sujet.pdf`](docs/sujet.pdf).
