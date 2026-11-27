package ma.enset.nn.decorator;

import ma.enset.nn.core.Layer;
import ma.enset.nn.core.Neuron;

import java.util.List;

/**
 * Decorator concret : normalisation par lot (batch normalization). Centre et
 * reduit les sorties de la couche decoree, puis propage les valeurs normalisees.
 */
public class BatchNormalizationDecorator extends LayerDecorator {

    private static final double EPSILON = 1e-8;

    public BatchNormalizationDecorator(Layer delegate) {
        super(delegate);
    }

    @Override
    public double[] forward(double[] inputs) {
        double[] outputs = super.forward(inputs);
        if (outputs.length == 0) {
            return outputs;
        }

        double mean = 0.0;
        for (double value : outputs) {
            mean += value;
        }
        mean /= outputs.length;

        double variance = 0.0;
        for (double value : outputs) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= outputs.length;

        double standardDeviation = Math.sqrt(variance + EPSILON);
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = (outputs[i] - mean) / standardDeviation;
        }

        List<Neuron> neurons = delegate.getNeurons();
        for (int i = 0; i < outputs.length; i++) {
            neurons.get(i).setOutput(outputs[i]);
        }
        return outputs;
    }

    @Override
    public String description() {
        return super.description() + " + BatchNorm";
    }
}
