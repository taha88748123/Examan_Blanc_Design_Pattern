package ma.enset.nn;

import ma.enset.nn.activation.ActivationFunction;
import ma.enset.nn.activation.LegacyActivationAdapter;
import ma.enset.nn.activation.LegacyArcTanSquashing;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.builder.NeuralNetworkBuilder;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.NeuralNetwork;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AdapterTest {

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
    }

    @Test
    void legacyFunctionShouldUseItsOwnInterface() {
        LegacyArcTanSquashing legacy = new LegacyArcTanSquashing();

        assertEquals(0.0, legacy.squash(0.0), 1e-9);
        assertEquals(0.5, legacy.squash(1.0), 1e-9);
        assertEquals("LegacyArcTan", legacy.getLabel());
    }

    @Test
    void adapterShouldExposeLegacyFunctionAsActivationFunction() {
        ActivationFunction adapter = new LegacyActivationAdapter(new LegacyArcTanSquashing());

        assertEquals(Math.atan(2.0) * 2.0 / Math.PI, adapter.apply(2.0), 1e-9);
        assertEquals("Adapter[LegacyArcTan]", adapter.name());
        assertTrue(adapter instanceof LegacyActivationAdapter);
    }

    @Test
    void adaptedFunctionShouldBeUsableInsideABuiltNetwork() {
        ActivationFunction adapter = new LegacyActivationAdapter(new LegacyArcTanSquashing());

        NeuralNetwork network = new NeuralNetworkBuilder()
                .name("adapter-network")
                .seed(5)
                .epochs(3)
                .inputLayer(2)
                .hiddenLayer(3, adapter)
                .outputLayer(1, new SigmoidActivation())
                .build();

        network.train();
        double[] output = network.predict(new double[]{0.3, 0.7});

        assertEquals(1, output.length);
        assertTrue(output[0] >= 0.0 && output[0] <= 1.0);
    }
}
