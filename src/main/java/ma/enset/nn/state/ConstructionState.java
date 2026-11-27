package ma.enset.nn.state;

import ma.enset.nn.core.NeuralNetwork;

/**
 * Etat initial : le reseau est en cours de construction. La prediction est
 * interdite tant que le reseau n'a pas ete entraine.
 */
public class ConstructionState implements NeuralNetworkState {

    @Override
    public NetworkStateType type() {
        return NetworkStateType.CONSTRUCTION;
    }

    @Override
    public String name() {
        return "Construction";
    }

    @Override
    public double[] predict(NeuralNetwork network, double[] input) {
        throw new IllegalStateException(
                "Le reseau est en construction : appelez train() ou fit() avant de predire");
    }

    @Override
    public void train(NeuralNetwork network) {
        network.transitionTo(new TrainingState(), "Debut de l'entrainement (Construction -> Training)");
        network.runTraining();
        network.transitionTo(new TrainedState(), "Entrainement termine (Training -> Trained)");
    }

    @Override
    public void fit(NeuralNetwork network) {
        network.transitionTo(new TrainingState(), "fit() : passage en execution via entrainement");
        network.runTraining();
        network.transitionTo(new TrainedState(), "fit() : le reseau est pret pour l'execution");
    }
}
