package mlAPI.math

/**
 * A trait identifying data points that are ready for training or prediction. These points are [[LabeledPoint]] and
 * [[UnlabeledPoint]] instances.
 */
trait LearningPoint extends Point
