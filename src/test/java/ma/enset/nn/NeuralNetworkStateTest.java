package ma.enset.nn;

import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.core.SimpleLayer;
import ma.enset.nn.state.NetworkStateType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralNetworkStateTest {

    private NeuralNetwork network;

    @BeforeEach
    void setUp() {
        NetworkConfig.getInstance().setEpochs(10);
        network = new NeuralNetwork("state-test", List.of(
                new SimpleLayer(LayerType.INPUT, 3, new IdentityActivation()),
                new SimpleLayer(LayerType.HIDDEN, 5, new SigmoidActivation(), new Random(1)),
                new SimpleLayer(LayerType.OUTPUT, 2, new SoftmaxActivation(), new Random(2))
        ));
    }

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
    }

    @Test
    void networkShouldStartInConstructionState() {
        assertEquals(NetworkStateType.CONSTRUCTION, network.getStateType());
        assertEquals("Construction", network.getStateName());
    }

    @Test
    void predictShouldBeForbiddenInConstructionState() {
        IllegalStateException error = assertThrows(IllegalStateException.class,
                () -> network.predict(new double[]{0.1, 0.2, 0.3}));
        assertTrue(error.getMessage().contains("construction"));
    }

    @Test
    void trainShouldMoveNetworkToTrainedState() {
        network.train();

        assertEquals(NetworkStateType.TRAINED, network.getStateType());
        assertEquals(10, network.getLossHistory().size());

        double[] output = network.predict(new double[]{0.2, 0.5, 0.8});
        assertEquals(2, output.length);
        assertEquals(1.0, output[0] + output[1], 1e-9);
    }

    @Test
    void fitShouldMoveNetworkFromConstructionToExecution() {
        network.fit();

        assertEquals(NetworkStateType.TRAINED, network.getStateType());
        assertTrue(network.getLossHistory().size() >= 10);
    }

    @Test
    void predictShouldRejectWrongInputSize() {
        network.train();
        assertThrows(IllegalArgumentException.class, () -> network.predict(new double[]{1.0}));
    }

    @Test
    void trainedNetworkCanBeRetrained() {
        network.train();
        int firstTrainingLosses = network.getLossHistory().size();
        network.train();

        assertEquals(NetworkStateType.TRAINED, network.getStateType());
        assertEquals(firstTrainingLosses + 10, network.getLossHistory().size());
    }
}
