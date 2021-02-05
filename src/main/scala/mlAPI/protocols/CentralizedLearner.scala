package mlAPI.protocols

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.math.UsablePoint
import mlAPI.parameters.utils.{ParameterDescriptor => QueryDescriptor}

@RemoteProxy
trait CentralizedLearner {

  @RemoteOp
  def forward(record: UsablePoint): Unit

  @RemoteOp
  def describe(qD: QueryDescriptor): Unit

}
