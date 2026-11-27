package ma.enset.nn.activation;

/**
 * Fonction d'activation ReLU : f(x) = max(0, x).
 */
public class ReLUActivation implements ActivationFunction {

    @Override
    public double apply(double input) {
        return Math.max(0.0, input);
    }

    @Override
    public String name() {
        return "ReLU";
    }
}
