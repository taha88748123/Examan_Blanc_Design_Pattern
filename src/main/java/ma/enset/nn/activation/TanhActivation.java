package ma.enset.nn.activation;

/**
 * Fonction d'activation tangente hyperbolique : f(x) = tanh(x).
 */
public class TanhActivation implements ActivationFunction {

    @Override
    public double apply(double input) {
        return Math.tanh(input);
    }

    @Override
    public String name() {
        return "Tanh";
    }
}
