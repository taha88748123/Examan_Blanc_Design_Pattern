package ma.enset.nn.aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * Aspect (exigence f) : journalisation de toutes les methodes du domaine.
 * Chaque appel est trace (methode, arguments, resultat, duree) dans
 * logs/neural-network.log.
 */
@Aspect
public class LoggingAspect {

    private static final String LOG_DIRECTORY = "logs";
    private static final String LOG_FILE = LOG_DIRECTORY + "/neural-network.log";
    private static final Logger LOGGER = createLogger();

    private static Logger createLogger() {
        try {
            Files.createDirectories(Path.of(LOG_DIRECTORY));
            Path logPath = Path.of(LOG_FILE);
            if (Files.exists(logPath) && Files.size(logPath) > 10 * 1024 * 1024) {
                Files.delete(logPath);
            }
            Logger logger = Logger.getLogger("ma.enset.nn.aop");
            logger.setUseParentHandlers(false);
            logger.setLevel(Level.ALL);
            FileHandler handler = new FileHandler(LOG_FILE, true);
            handler.setFormatter(new SimpleFormatter());
            logger.addHandler(handler);
            return logger;
        } catch (IOException exception) {
            throw new IllegalStateException("Impossible d'initialiser le journal des aspects", exception);
        }
    }

    public static Logger getLogger() {
        return LOGGER;
    }

    @Around("execution(* ma.enset.nn..*.*(..)) && !within(ma.enset.nn.aspect..*) "
            + "&& !within(ma.enset.nn..*Test*)")
    public Object logMethodCall(ProceedingJoinPoint joinPoint) throws Throwable {
        long start = System.nanoTime();
        try {
            Object result = joinPoint.proceed();
            long durationMs = (System.nanoTime() - start) / 1_000_000;
            LOGGER.info(String.format("%s | args=%s | result=%s | %d ms",
                    joinPoint.getSignature().toShortString(),
                    format(joinPoint.getArgs()),
                    format(new Object[]{result}),
                    durationMs));
            return result;
        } catch (Throwable throwable) {
            LOGGER.warning(String.format("%s | args=%s | exception=%s",
                    joinPoint.getSignature().toShortString(),
                    format(joinPoint.getArgs()),
                    throwable.getClass().getSimpleName() + ": " + throwable.getMessage()));
            throw throwable;
        }
    }

    private String format(Object[] values) {
        if (values == null) {
            return "[]";
        }
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                builder.append(", ");
            }
            Object value = values[i];
            if (value instanceof double[] array) {
                builder.append(Arrays.toString(array));
            } else if (value instanceof Object[] array) {
                builder.append(Arrays.toString(array));
            } else {
                builder.append(value);
            }
        }
        return builder.append(']').toString();
    }
}
