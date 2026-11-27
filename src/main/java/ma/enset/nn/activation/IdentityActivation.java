package ma.enset.nn.activation;

/**
 * Fonction d'activation identite : f(x) = x.
 */
public class IdentityActivation implements ActivationFunction {

    @Override
    public double apply(double input) {
        return input;
    }

    @Override
    public String name() {
        return "Identity";
    }
}
