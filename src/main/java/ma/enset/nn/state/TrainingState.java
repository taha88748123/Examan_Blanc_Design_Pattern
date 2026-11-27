package ma.enset.nn.state;

import ma.enset.nn.core.NeuralNetwork;

/**
 * Etat pendant l'entrainement : ni train() ni predict() ne sont autorises.
 */
public class TrainingState implements NeuralNetworkState {

    @Override
    public NetworkStateType type() {
        return NetworkStateType.TRAINING;
    }

    @Override
    public String name() {
        return "Training";
    }

    @Override
    public double[] predict(NeuralNetwork network, double[] input) {
        throw new IllegalStateException(
                "Le reseau est en cours d'entrainement : predict() indisponible");
    }

    @Override
    public void train(NeuralNetwork network) {
        throw new IllegalStateException("Un entrainement est deja en cours");
    }

    @Override
    public void fit(NeuralNetwork network) {
        throw new IllegalStateException("Un entrainement est deja en cours");
    }
}
