package ma.enset.nn.core;

import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.observer.NetworkEvent;
import ma.enset.nn.observer.NetworkObserver;
import ma.enset.nn.observer.NetworkSubject;
import ma.enset.nn.state.ConstructionState;
import ma.enset.nn.state.NetworkStateType;
import ma.enset.nn.state.NeuralNetworkState;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Contexte du reseau de neurones. Il delegue son comportement a un etat
 * (State), notifie ses observateurs (Observer) et expose les operations
 * train(), fit() et predict() demandees par l'enonce.
 */
public class NeuralNetwork implements NetworkSubject {

    private final String name;
    private final List<Layer> layers;
    private final NetworkConfig config;
    private final Random random;
    private final List<NetworkObserver> observers = new CopyOnWriteArrayList<>();
    private final List<Double> lossHistory = new ArrayList<>();

    private NeuralNetworkState state = new ConstructionState();

    public NeuralNetwork(String name, List<Layer> layers) {
        this(name, layers, NetworkConfig.getInstance(), new Random(NetworkConfig.getInstance().getRandomSeed()));
    }

    public NeuralNetwork(String name, List<Layer> layers, NetworkConfig config, Random random) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("Le nom du reseau est obligatoire");
        }
        this.name = name;
        this.config = Objects.requireNonNull(config, "config");
        this.random = Objects.requireNonNull(random, "random");
        this.layers = validateAndConnect(layers);
    }

    private List<Layer> validateAndConnect(List<Layer> sourceLayers) {
        Objects.requireNonNull(sourceLayers, "layers");
        if (sourceLayers.size() < 2) {
            throw new IllegalArgumentException("Un reseau doit contenir au moins une couche d'entree et une couche de sortie");
        }
        List<Layer> copy = new ArrayList<>(sourceLayers);
        if (copy.get(0).type() != LayerType.INPUT) {
            throw new IllegalArgumentException("La premiere couche doit etre une couche d'entree");
        }
        if (copy.get(copy.size() - 1).type() != LayerType.OUTPUT) {
            throw new IllegalArgumentException("La derniere couche doit etre une couche de sortie");
        }
        for (int i = 1; i < copy.size() - 1; i++) {
            if (copy.get(i).type() != LayerType.HIDDEN) {
                throw new IllegalArgumentException("Les couches intermediaires doivent etre des couches cachees");
            }
        }
        for (int i = 0; i < copy.size() - 1; i++) {
            copy.get(i).connectTo(copy.get(i + 1));
        }
        return Collections.unmodifiableList(copy);
    }

    public double[] predict(double[] input) {
        Objects.requireNonNull(input, "input");
        return state.predict(this, input);
    }

    public void train() {
        state.train(this);
    }

    public void fit() {
        state.fit(this);
    }

    public double[] forward(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    public void runTraining() {
        int epochs = config.getEpochs();
        double learningRate = config.getLearningRate();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] sample = randomInput();
            double[] output = forward(sample);
            double loss = 0.0;
            for (double value : output) {
                loss += Math.pow(1.0 - value, 2);
            }
            lossHistory.add(loss / output.length);
            nudgeWeights(learningRate);
        }
    }

    private double[] randomInput() {
        int size = layers.get(0).neuronCount();
        double[] sample = new double[size];
        for (int i = 0; i < size; i++) {
            sample[i] = random.nextDouble();
        }
        return sample;
    }

    private void nudgeWeights(double learningRate) {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                for (Connection connection : neuron.getIncomingConnections()) {
                    double updated = connection.getWeight()
                            + learningRate * (random.nextDouble() - 0.5) * 0.01;
                    connection.setWeight(Math.max(0.0, Math.min(1.0, updated)));
                }
            }
        }
    }

    public void transitionTo(NeuralNetworkState newState, String message) {
        this.state = Objects.requireNonNull(newState, "newState");
        notifyObservers(new NetworkEvent(name, newState.type(), message, Instant.now()));
    }

    @Override
    public void addObserver(NetworkObserver observer) {
        observers.add(Objects.requireNonNull(observer, "observer"));
    }

    @Override
    public void removeObserver(NetworkObserver observer) {
        observers.remove(observer);
    }

    @Override
    public void notifyObservers(NetworkEvent event) {
        for (NetworkObserver observer : observers) {
            observer.onNetworkEvent(event);
        }
    }

    public String getName() {
        return name;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public NetworkConfig getConfig() {
        return config;
    }

    public NetworkStateType getStateType() {
        return state.type();
    }

    public String getStateName() {
        return state.name();
    }

    public NeuralNetworkState getState() {
        return state;
    }

    public List<Double> getLossHistory() {
        return Collections.unmodifiableList(lossHistory);
    }
}
