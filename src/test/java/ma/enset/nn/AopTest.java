package ma.enset.nn;

import ma.enset.nn.aspect.LoggingAspect;
import ma.enset.nn.aspect.SecurityContext;
import ma.enset.nn.builder.NeuralNetworkBuilder;
import ma.enset.nn.config.NetworkConfig;
import ma.enset.nn.core.NeuralNetwork;
import ma.enset.nn.state.NetworkStateType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Handler;
import java.util.logging.LogRecord;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AopTest {

    private NeuralNetwork network;

    @BeforeEach
    void setUp() {
        NetworkConfig.getInstance().setEpochs(3);
        network = new NeuralNetworkBuilder()
                .name("aop-test")
                .seed(11)
                .inputLayer(2)
                .hiddenLayer(3)
                .outputLayer(1)
                .build();
    }

    @AfterEach
    void tearDown() {
        NetworkConfig.getInstance().resetDefaults();
        SecurityContext.clear();
    }

    @Test
    void loggingAspectShouldTraceMethodCalls() {
        List<LogRecord> records = Collections.synchronizedList(new ArrayList<>());
        Handler handler = new Handler() {
            @Override
            public void publish(LogRecord record) {
                records.add(record);
            }

            @Override
            public void flush() {
            }

            @Override
            public void close() {
            }
        };
        LoggingAspect.getLogger().addHandler(handler);
        try {
            network.train();
        } finally {
            LoggingAspect.getLogger().removeHandler(handler);
        }

        assertTrue(records.stream().anyMatch(record -> record.getMessage().contains("NeuralNetwork.train")));
        assertTrue(records.stream().anyMatch(record -> record.getMessage().contains("TransitionTo")
                || record.getMessage().contains("transitionTo")));
        assertTrue(Files.exists(Path.of("logs", "neural-network.log")));
    }

    @Test
    void guestShouldNotBeAllowedToTrain() {
        SecurityContext.setRole(SecurityContext.GUEST_ROLE);

        SecurityException exception = assertThrows(SecurityException.class, network::train);

        assertTrue(exception.getMessage().contains("Acces refuse"));
        assertEquals(NetworkStateType.CONSTRUCTION, network.getStateType());
    }

    @Test
    void guestShouldNotBeAllowedToPredict() {
        network.train();
        SecurityContext.setRole(SecurityContext.GUEST_ROLE);

        assertThrows(SecurityException.class, () -> network.predict(new double[]{0.1, 0.9}));
    }

    @Test
    void adminShouldBeAllowedToTrain() {
        SecurityContext.setRole(SecurityContext.ADMIN_ROLE);

        network.train();

        assertEquals(NetworkStateType.TRAINED, network.getStateType());
    }
}
