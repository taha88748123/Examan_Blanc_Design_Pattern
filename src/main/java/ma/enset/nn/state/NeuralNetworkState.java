package ma.enset.nn.state;

import ma.enset.nn.core.NeuralNetwork;

/**
 * State (exigence c) : chaque etat encapsule le comportement autorise du reseau.
 */
public interface NeuralNetworkState {

    NetworkStateType type();

    String name();

    double[] predict(NeuralNetwork network, double[] input);

    void train(NeuralNetwork network);

    void fit(NeuralNetwork network);
}
