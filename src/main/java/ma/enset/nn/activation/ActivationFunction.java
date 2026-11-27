package ma.enset.nn.activation;

/**
 * Strategy (exigence a) : famille de fonctions d'activation encapsulees et
 * interchangeables au niveau des neurones.
 */
public interface ActivationFunction {

    double apply(double input);

    String name();

    default boolean isVectorBased() {
        return false;
    }

    default double[] apply(double[] inputs) {
        double[] result = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            result[i] = apply(inputs[i]);
        }
        return result;
    }
}
