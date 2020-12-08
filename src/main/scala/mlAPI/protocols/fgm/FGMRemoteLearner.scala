package mlAPI.protocols.fgm

import BipartiteTopologyAPI.annotations.{RemoteOp, RemoteProxy}
import mlAPI.parameters.ParameterDescriptor

@RemoteProxy
trait FGMRemoteLearner {

  @RemoteOp
  def reset(model: ParameterDescriptor): Unit

  @RemoteOp
  def sendLocalDrift(): Unit

  @RemoteOp
  def requestZeta(): Unit

  @RemoteOp
  def receiveQuantum(quantum: Quantum): Unit

}
