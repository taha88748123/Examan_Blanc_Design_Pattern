package ma.enset.nn.aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;

import java.util.logging.Logger;

/**
 * Aspect (exigence g) : securise les methodes annotees @Secured en verifiant
 * le role de l'utilisateur courant.
 */
@Aspect
public class SecurityAspect {

    private static final Logger LOGGER = LoggingAspect.getLogger();

    @Around("execution(@ma.enset.nn.aspect.Secured * ma.enset.nn..*(..)) && @annotation(secured)")
    public Object checkAccess(ProceedingJoinPoint joinPoint, Secured secured) throws Throwable {
        String requiredRole = secured.value();
        if (!SecurityContext.hasRole(requiredRole)) {
            String message = String.format("Acces refuse a %s : role requis '%s', role courant '%s'",
                    joinPoint.getSignature().toShortString(), requiredRole, SecurityContext.getRole());
            LOGGER.warning("[SECURITE] " + message);
            throw new SecurityException(message);
        }
        return joinPoint.proceed();
    }
}
