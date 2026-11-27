package ma.enset.nn.activation;

/**
 * Fonction d'activation Sigmoide : f(x) = 1 / (1 + e^-x).
 */
public class SigmoidActivation implements ActivationFunction {

    @Override
    public double apply(double input) {
        return 1.0 / (1.0 + Math.exp(-input));
    }

    @Override
    public String name() {
        return "Sigmoid";
    }
}
