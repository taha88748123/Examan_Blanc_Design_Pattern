package ma.enset.nn;

import ma.enset.nn.activation.IdentityActivation;
import ma.enset.nn.activation.SigmoidActivation;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.LayerType;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.core.SimpleLayer;
import ma.enset.nn.observer.Dashboard;
import ma.enset.nn.state.NetworkStateType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ObserverTest {

    private NeuralNetwork network;
    private Dashboard dashboard;

    @BeforeEach
    void setUp() {
        NetworkConfig.getInstance().setEpochs(3);
        network = new NeuralNetwork("observer-test", List.of(
                new SimpleLayer(LayerType.INPUT, 2, new IdentityActivation()),
                new SimpleLayer(LayerType.HIDDEN, 3, new SigmoidActivation(), new Random(7)),
                new SimpleLayer(LayerType.OUTPUT, 1, new SigmoidActivation(), new Random(8))
        ));
        dashboard = new Dashboard();
        network.addObserver(dashboard);
    }

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
    }

    @Test
    void dashboardShouldBeNotifiedOnEachStateChange() {
        network.train();

        assertEquals(1, dashboard.count(NetworkStateType.TRAINING));
        assertEquals(1, dashboard.count(NetworkStateType.TRAINED));
        assertEquals(2, dashboard.getHistory().size());
        assertEquals(NetworkStateType.TRAINING, dashboard.getHistory().get(0).state());
        assertEquals(NetworkStateType.TRAINED, dashboard.getHistory().get(1).state());
        assertEquals(2, dashboard.getPrintedLines().size());
    }

    @Test
    void removedObserverShouldNoLongerBeNotified() {
        network.train();
        int eventsBefore = dashboard.getHistory().size();

        network.removeObserver(dashboard);
        network.train();

        assertEquals(eventsBefore, dashboard.getHistory().size());
    }

    @Test
    void summaryShouldExposeAggregatedStatistics() {
        network.train();
        network.train();

        assertTrue(dashboard.summary().contains("TRAINED=2"));
        assertEquals(2, dashboard.count(NetworkStateType.TRAINING));
        assertEquals(2, dashboard.count(NetworkStateType.TRAINED));
    }
}
