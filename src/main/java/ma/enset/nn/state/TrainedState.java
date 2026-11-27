package ma.enset.nn.state;

import ma.enset.nn.core.NeuralNetwork;

import java.util.Objects;

/**
 * Etat final : le reseau est entraine, predict() est autorise. Un nouvel
 * entrainement reste possible (Trained -> Training -> Trained).
 */
public class TrainedState implements NeuralNetworkState {

    @Override
    public NetworkStateType type() {
        return NetworkStateType.TRAINED;
    }

    @Override
    public String name() {
        return "Trained";
    }

    @Override
    public double[] predict(NeuralNetwork network, double[] input) {
        Objects.requireNonNull(input, "input");
        int expected = network.getLayers().get(0).neuronCount();
        if (input.length != expected) {
            throw new IllegalArgumentException(
                    "Taille d'entree invalide : attendu " + expected + ", recu " + input.length);
        }
        return network.forward(input);
    }

    @Override
    public void train(NeuralNetwork network) {
        network.transitionTo(new TrainingState(), "Re-entrainement demande (Trained -> Training)");
        network.runTraining();
        network.transitionTo(new TrainedState(), "Re-entrainement termine (Training -> Trained)");
    }

    @Override
    public void fit(NeuralNetwork network) {
        train(network);
    }
}
