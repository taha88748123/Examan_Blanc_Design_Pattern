package ma.enset.nn.decorator;

import ma.enset.nn.core.Layer;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.Neuron;

import java.util.List;
import java.util.Objects;

/**
 * Decorator (exigence e) : base commune des couches decorees. Le decorateur
 * reste une Layer et delegue le comportement de base a la couche enveloppee.
 */
public abstract class LayerDecorator implements Layer {

    protected final Layer delegate;

    protected LayerDecorator(Layer delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    @Override
    public double[] forward(double[] inputs) {
        return delegate.forward(inputs);
    }

    @Override
    public void connectTo(Layer nextLayer) {
        delegate.connectTo(nextLayer);
    }

    @Override
    public int neuronCount() {
        return delegate.neuronCount();
    }

    @Override
    public LayerType type() {
        return delegate.type();
    }

    @Override
    public String description() {
        return delegate.description();
    }

    @Override
    public List<Neuron> getNeurons() {
        return delegate.getNeurons();
    }

    public Layer getDelegate() {
        return delegate;
    }
}
