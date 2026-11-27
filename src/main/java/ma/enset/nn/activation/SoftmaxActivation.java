package ma.enset.nn.activation;

/**
 * Fonction d'activation Softmax. Contrairement aux autres fonctions, elle
 * s'applique a l'ensemble des neurones d'une couche : la somme des sorties vaut 1.
 */
public class SoftmaxActivation implements ActivationFunction {

    @Override
    public double apply(double input) {
        return 1.0;
    }

    @Override
    public boolean isVectorBased() {
        return true;
    }

    @Override
    public double[] apply(double[] inputs) {
        if (inputs.length == 0) {
            return new double[0];
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double input : inputs) {
            max = Math.max(max, input);
        }
        double[] exponentials = new double[inputs.length];
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            exponentials[i] = Math.exp(inputs[i] - max);
            sum += exponentials[i];
        }
        for (int i = 0; i < inputs.length; i++) {
            exponentials[i] /= sum;
        }
        return exponentials;
    }

    @Override
    public String name() {
        return "Softmax";
    }
}
