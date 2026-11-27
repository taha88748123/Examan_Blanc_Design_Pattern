package ma.enset.nn.aspect;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marque une methode comme protegee par le SecurityAspect (exigence g).
 * La valeur correspond au role requis pour executer la methode.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.TYPE})
public @interface Secured {

    String value() default "DATA_SCIENTIST";
}
