package ma.enset.nn.decorator;

import ma.enset.nn.core.Layer;
import ma.enset.nn.core.Neuron;

import java.util.List;
import java.util.Random;

/**
 * Decorator concret : dropout. Desactive aleatoirement une proportion de
 * neurones (regularisation) et propage le masque aux neurones de la couche.
 */
public class DropoutDecorator extends LayerDecorator {

    private final double rate;
    private final Random random;

    public DropoutDecorator(Layer delegate, double rate) {
        this(delegate, rate, new Random());
    }

    public DropoutDecorator(Layer delegate, double rate, Random random) {
        super(delegate);
        if (rate < 0.0 || rate > 1.0) {
            throw new IllegalArgumentException("Le taux de dropout doit etre compris entre 0 et 1");
        }
        this.rate = rate;
        this.random = java.util.Objects.requireNonNull(random, "random");
    }

    @Override
    public double[] forward(double[] inputs) {
        double[] outputs = super.forward(inputs);
        if (rate == 0.0) {
            return outputs;
        }
        double scale = rate == 1.0 ? 0.0 : 1.0 / (1.0 - rate);
        for (int i = 0; i < outputs.length; i++) {
            if (rate == 1.0 || random.nextDouble() < rate) {
                outputs[i] = 0.0;
            } else {
                outputs[i] *= scale;
            }
        }
        propagateToNeurons(outputs);
        return outputs;
    }

    private void propagateToNeurons(double[] outputs) {
        List<Neuron> neurons = delegate.getNeurons();
        for (int i = 0; i < outputs.length; i++) {
            neurons.get(i).setOutput(outputs[i]);
        }
    }

    @Override
    public String description() {
        return super.description() + " + Dropout(p=" + rate + ")";
    }

    public double getRate() {
        return rate;
    }
}
