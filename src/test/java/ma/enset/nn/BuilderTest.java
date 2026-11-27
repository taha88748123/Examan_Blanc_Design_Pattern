package ma.enset.nn;

import ma.enset.nn.builder.NeuralNetworkBuilder;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.Connection;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.observer.Dashboard;
import ma.enset.nn.state.NetworkStateType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BuilderTest {

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
    }

    @Test
    void builderShouldCreateNetworkWithExpectedArchitecture() {
        NeuralNetwork network = new NeuralNetworkBuilder()
                .name("builder-test")
                .seed(10)
                .inputLayer(3)
                .hiddenLayer(5)
                .hiddenLayer(4)
                .outputLayer(2)
                .build();

        assertEquals("builder-test", network.getName());
        assertEquals(4, network.getLayers().size());
        assertEquals(3, network.getLayers().get(0).neuronCount());
        assertEquals(5, network.getLayers().get(1).neuronCount());
        assertEquals(4, network.getLayers().get(2).neuronCount());
        assertEquals(2, network.getLayers().get(3).neuronCount());
        assertEquals(LayerType.INPUT, network.getLayers().get(0).type());
        assertEquals(LayerType.OUTPUT, network.getLayers().get(3).type());
        assertEquals(NetworkStateType.CONSTRUCTION, network.getStateType());
    }

    @Test
    void builderShouldRejectIncompleteNetwork() {
        assertThrows(IllegalStateException.class, () -> new NeuralNetworkBuilder()
                .inputLayer(3)
                .build());
    }

    @Test
    void builderShouldAttachObservers() {
        Dashboard dashboard = new Dashboard();
        NeuralNetwork network = new NeuralNetworkBuilder()
                .name("observed")
                .epochs(3)
                .inputLayer(2)
                .hiddenLayer(3)
                .outputLayer(1)
                .observer(dashboard)
                .build();

        network.train();

        assertEquals(1, dashboard.count(NetworkStateType.TRAINING));
        assertEquals(1, dashboard.count(NetworkStateType.TRAINED));
    }

    @Test
    void builderShouldSetEpochsAndLearningRate() {
        NeuralNetwork network = new NeuralNetworkBuilder()
                .epochs(7)
                .learningRate(0.2)
                .inputLayer(2)
                .hiddenLayer(2)
                .outputLayer(1)
                .build();

        assertEquals(7, network.getConfig().getEpochs());
        assertEquals(0.2, network.getConfig().getLearningRate(), 0.0);

        network.train();
        assertEquals(7, network.getLossHistory().size());
    }

    @Test
    void sameSeedShouldProduceSameInitialWeights() {
        NeuralNetwork first = new NeuralNetworkBuilder()
                .seed(123)
                .inputLayer(2)
                .hiddenLayer(3)
                .outputLayer(1)
                .build();
        NeuralNetwork second = new NeuralNetworkBuilder()
                .seed(123)
                .inputLayer(2)
                .hiddenLayer(3)
                .outputLayer(1)
                .build();

        Connection firstConnection = first.getLayers().get(1).getNeurons().get(0).getIncomingConnections().get(0);
        Connection secondConnection = second.getLayers().get(1).getNeurons().get(0).getIncomingConnections().get(0);

        assertEquals(firstConnection.getWeight(), secondConnection.getWeight(), 0.0);
    }
}
