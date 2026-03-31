use crux.core
use crux.errors

pub fn allreduce_sum(stream: *mut Stream, src: View, dst: View) -> Result[*mut Event, SubstrateError]:
    let _ = stream
    let _ = src
    let _ = dst
    Err(.Unsupported("collectives are not implemented yet"))

pub fn allgather(stream: *mut Stream, src: View, dst: View) -> Result[*mut Event, SubstrateError]:
    let _ = stream
    let _ = src
    let _ = dst
    Err(.Unsupported("collectives are not implemented yet"))

pub fn broadcast(stream: *mut Stream, src: View, dst: View) -> Result[*mut Event, SubstrateError]:
    let _ = stream
    let _ = src
    let _ = dst
    Err(.Unsupported("collectives are not implemented yet"))

pub fn reduce_scatter_sum(stream: *mut Stream, src: View, dst: View) -> Result[*mut Event, SubstrateError]:
    let _ = stream
    let _ = src
    let _ = dst
    Err(.Unsupported("collectives are not implemented yet"))
