package ma.enset.nn;

import ma.enset.nn.config.NetworkConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NetworkConfigTest {

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
    }

    @Test
    void singletonShouldAlwaysReturnSameInstance() {
        assertSame(NetworkConfig.getInstance(), NetworkConfig.getInstance());
    }

    @Test
    void configurationShouldBeMutableWithValidation() {
        NetworkConfig config = NetworkConfig.getInstance();
        config.setEpochs(120);
        config.setLearningRate(0.5);

        assertEquals(120, config.getEpochs());
        assertEquals(0.5, config.getLearningRate(), 0.0);

        assertThrows(IllegalArgumentException.class, () -> config.setEpochs(0));
        assertThrows(IllegalArgumentException.class, () -> config.setLearningRate(0.0));
        assertThrows(IllegalArgumentException.class, () -> config.setLearningRate(1.5));
    }

    @Test
    void resetShouldRestoreDefaults() {
        NetworkConfig config = NetworkConfig.getInstance();
        config.setEpochs(5);
        config.setLearningRate(0.9);

        config.resetDefaults();

        assertEquals(50, config.getEpochs());
        assertEquals(0.1, config.getLearningRate(), 0.0);
    }
}
