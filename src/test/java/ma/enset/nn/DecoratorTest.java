package ma.enset.nn;

import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.builder.NeuralNetworkBuilder;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.Layer;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.core.SimpleLayer;
import ma.enset.nn.decorator.BatchNormalizationDecorator;
import ma.enset.nn.decorator.DropoutDecorator;
import ma.enset.nn.decorator.LayerDecorator;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecoratorTest {

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
    }

    private SimpleLayer connectedHiddenLayer(int neurons, long seed) {
        SimpleLayer input = new SimpleLayer(LayerType.INPUT, 3, new IdentityActivation());
        SimpleLayer hidden = new SimpleLayer(LayerType.HIDDEN, neurons, new SigmoidActivation(), new Random(seed));
        input.connectTo(hidden);
        input.forward(new double[]{0.2, 0.5, 0.9});
        return hidden;
    }

    @Test
    void dropoutWithRateOneShouldDisableAllOutputs() {
        SimpleLayer hidden = connectedHiddenLayer(4, 1);
        LayerDecorator dropout = new DropoutDecorator(hidden, 1.0, new Random(2));

        double[] outputs = dropout.forward(new double[]{0.2, 0.5, 0.9});

        for (double output : outputs) {
            assertEquals(0.0, output, 0.0);
        }
        for (int i = 0; i < outputs.length; i++) {
            assertEquals(0.0, hidden.getNeurons().get(i).getOutput(), 0.0);
        }
        assertTrue(dropout.description().contains("Dropout"));
    }

    @Test
    void dropoutWithRateZeroShouldKeepDelegateOutputs() {
        SimpleLayer hidden = connectedHiddenLayer(4, 3);
        double[] reference = hidden.forward(new double[]{0.2, 0.5, 0.9});

        DropoutDecorator dropout = new DropoutDecorator(hidden, 0.0, new Random(4));
        double[] outputs = dropout.forward(new double[]{0.2, 0.5, 0.9});

        assertArrayEquals(reference, outputs, 1e-12);
    }

    @Test
    void dropoutShouldRejectInvalidRate() {
        SimpleLayer hidden = connectedHiddenLayer(3, 5);
        assertThrows(IllegalArgumentException.class, () -> new DropoutDecorator(hidden, -0.1));
        assertThrows(IllegalArgumentException.class, () -> new DropoutDecorator(hidden, 1.5));
    }

    @Test
    void batchNormalizationShouldCenterAndReduceOutputs() {
        SimpleLayer hidden = connectedHiddenLayer(5, 6);
        BatchNormalizationDecorator batchNorm = new BatchNormalizationDecorator(hidden);

        double[] outputs = batchNorm.forward(new double[]{0.2, 0.5, 0.9});

        double mean = 0.0;
        for (double value : outputs) {
            mean += value;
        }
        mean /= outputs.length;
        assertEquals(0.0, mean, 1e-7);

        double variance = 0.0;
        for (double value : outputs) {
            variance += Math.pow(value - mean, 2);
        }
        variance /= outputs.length;
        assertEquals(1.0, variance, 1e-4);
        assertTrue(batchNorm.description().contains("BatchNorm"));
    }

    @Test
    void decoratorsShouldBeComposableAndUsableInANetwork() {
        Layer hidden = new SimpleLayer(LayerType.HIDDEN, 4, new SigmoidActivation(), new Random(7));
        Layer decorated = new DropoutDecorator(new BatchNormalizationDecorator(hidden), 0.5, new Random(8));

        NeuralNetwork network = new NeuralNetworkBuilder()
                .name("decorated")
                .seed(9)
                .epochs(3)
                .inputLayer(3, new IdentityActivation())
                .layer(decorated)
                .outputLayer(2, new SoftmaxActivation())
                .build();

        network.train();
        double[] output = network.predict(new double[]{0.1, 0.2, 0.3});

        assertEquals(2, output.length);
        assertEquals(1.0, output[0] + output[1], 1e-9);
        assertTrue(network.getLayers().get(1).description().contains("BatchNorm"));
        assertTrue(network.getLayers().get(1).description().contains("Dropout"));
    }
}
