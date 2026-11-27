package ma.enset.nn;

import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.core.Connection;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.core.Neuron;
import ma.enset.nn.core.SimpleLayer;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CoreModelTest {

    @Test
    void connectionWeightShouldStayBetweenZeroAndOne() {
        Neuron source = new Neuron(0, new IdentityActivation());
        Neuron target = new Neuron(1, new IdentityActivation());

        Connection connection = new Connection(source, target, 0.5);
        assertEquals(0.5, connection.getWeight(), 0.0);

        assertThrows(IllegalArgumentException.class, () -> new Connection(source, target, -0.1));
        assertThrows(IllegalArgumentException.class, () -> new Connection(source, target, 1.1));
    }

    @Test
    void neuronShouldComputeWeightedSumAndApplyActivation() {
        Neuron sourceA = new Neuron(0, new IdentityActivation());
        sourceA.setOutput(1.0);
        Neuron sourceB = new Neuron(1, new IdentityActivation());
        sourceB.setOutput(0.5);

        Neuron target = new Neuron(2, 0.1, new SigmoidActivation());
        target.addIncomingConnection(new Connection(sourceA, target, 0.5));
        target.addIncomingConnection(new Connection(sourceB, target, 1.0));

        assertEquals(1.1, target.computeSum(), 1e-9);
        assertEquals(1.0 / (1.0 + Math.exp(-1.1)), target.forward(), 1e-9);
        assertEquals(2, target.getIncomingConnections().size());
    }

    @Test
    void inputLayerShouldReturnInputVectorAndValidateSize() {
        SimpleLayer inputLayer = new SimpleLayer(LayerType.INPUT, 3, new IdentityActivation());

        double[] inputs = {0.2, 0.4, 0.6};
        assertArrayEquals(inputs, inputLayer.forward(inputs), 0.0);
        assertThrows(IllegalArgumentException.class, () -> inputLayer.forward(new double[]{1.0}));
    }

    @Test
    void networkShouldValidateLayerArchitecture() {
        assertThrows(IllegalArgumentException.class, () -> new NeuralNetwork("invalid",
                List.of(new SimpleLayer(LayerType.INPUT, 3, new IdentityActivation()))));

        assertThrows(IllegalArgumentException.class, () -> new NeuralNetwork("invalid",
                List.of(new SimpleLayer(LayerType.HIDDEN, 3, new IdentityActivation()),
                        new SimpleLayer(LayerType.OUTPUT, 2, new IdentityActivation()))));

        assertThrows(IllegalArgumentException.class, () -> new NeuralNetwork("invalid",
                List.of(new SimpleLayer(LayerType.INPUT, 3, new IdentityActivation()),
                        new SimpleLayer(LayerType.HIDDEN, 2, new IdentityActivation()))));
    }

    @Test
    void hiddenAndOutputLayersShouldBeFullyConnected() {
        NeuralNetwork network = new NeuralNetwork("fully-connected", List.of(
                new SimpleLayer(LayerType.INPUT, 3, new IdentityActivation()),
                new SimpleLayer(LayerType.HIDDEN, 5, new SigmoidActivation(), new Random(3)),
                new SimpleLayer(LayerType.OUTPUT, 2, new SigmoidActivation(), new Random(4))
        ));

        assertEquals(3, network.getLayers().get(1).getNeurons().get(0).getIncomingConnections().size());
        assertEquals(5, network.getLayers().get(2).getNeurons().get(0).getIncomingConnections().size());
        assertEquals(5, network.getLayers().get(1).neuronCount());
        assertEquals("InputLayer[3] (Identity)", network.getLayers().get(0).description());
    }
}
