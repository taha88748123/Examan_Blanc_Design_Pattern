package ma.enset.nn.builder;

import ma.enset.nn.activation.ActivationFunction;
import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.activation.SoftmaxActivation;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.Layer;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.core.SimpleLayer;
import ma.enset.nn.observer.NetworkObserver;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Builder (amelioration) : construction fluide et lisible d'un reseau
 * entierement connecte, avec injection d'observateurs et graine deterministe.
 */
public class NeuralNetworkBuilder {

    private final List<Layer> layers = new ArrayList<>();
    private final List<NetworkObserver> observers = new ArrayList<>();
    private String name = "neural-network";
    private long seed = NetworkConfig.getInstance().getRandomSeed();
    private int epochs = -1;
    private double learningRate = -1.0;

    public NeuralNetworkBuilder name(String name) {
        this.name = Objects.requireNonNull(name, "name");
        return this;
    }

    public NeuralNetworkBuilder seed(long seed) {
        this.seed = seed;
        return this;
    }

    public NeuralNetworkBuilder epochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    public NeuralNetworkBuilder learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuralNetworkBuilder inputLayer(int neuronCount) {
        return inputLayer(neuronCount, new IdentityActivation());
    }

    public NeuralNetworkBuilder inputLayer(int neuronCount, ActivationFunction activation) {
        layers.add(createLayer(LayerType.INPUT, neuronCount, activation));
        return this;
    }

    public NeuralNetworkBuilder hiddenLayer(int neuronCount) {
        return hiddenLayer(neuronCount, new SigmoidActivation());
    }

    public NeuralNetworkBuilder hiddenLayer(int neuronCount, ActivationFunction activation) {
        layers.add(createLayer(LayerType.HIDDEN, neuronCount, activation));
        return this;
    }

    public NeuralNetworkBuilder outputLayer(int neuronCount) {
        return outputLayer(neuronCount, new SoftmaxActivation());
    }

    public NeuralNetworkBuilder outputLayer(int neuronCount, ActivationFunction activation) {
        layers.add(createLayer(LayerType.OUTPUT, neuronCount, activation));
        return this;
    }

    public NeuralNetworkBuilder layer(Layer layer) {
        layers.add(Objects.requireNonNull(layer, "layer"));
        return this;
    }

    public NeuralNetworkBuilder observer(NetworkObserver observer) {
        observers.add(Objects.requireNonNull(observer, "observer"));
        return this;
    }

    public NeuralNetwork build() {
        if (layers.size() < 2) {
            throw new IllegalStateException(
                    "Un reseau doit contenir au moins une couche d'entree et une couche de sortie");
        }
        NetworkConfig config = NetworkConfig.getInstance();
        if (epochs > 0) {
            config.setEpochs(epochs);
        }
        if (learningRate > 0.0) {
            config.setLearningRate(learningRate);
        }
        NeuralNetwork network = new NeuralNetwork(name, layers, config, new Random(seed));
        observers.forEach(network::addObserver);
        return network;
    }

    private SimpleLayer createLayer(LayerType type, int neuronCount, ActivationFunction activation) {
        return new SimpleLayer(type, neuronCount, activation, new Random(seed + layers.size()));
    }
}
