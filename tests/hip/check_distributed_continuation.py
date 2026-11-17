import copy,json,os,torch
from hip_orb import distributed_train as dt
from hip_orb.packed_consistency import packed_consistency_update
original=dt._distributed_consistency_update
verified=False
def verify(model,optimizer,batch,*args,**kwargs):
 global verified
 if not verified:
  verified=True
  sample=dt._slice_batch(batch,list(range(min(2,len(batch['sizes'])))))
  head=copy.deepcopy(model.head.state_dict()); state=copy.deepcopy(optimizer.state_dict())
  model.head.checkpoint_pair_activations=True
  expected_loss,_=packed_consistency_update(model,optimizer,sample,*args,**kwargs)
  expected_grads=[None if p.grad is None else p.grad.clone() for p in model.head.parameters()]
  expected=copy.deepcopy(model.head.state_dict()); expected_opt=copy.deepcopy(optimizer.state_dict())
  model.head.load_state_dict(head);optimizer.load_state_dict(state)
  from hip_orb.mixed_precision import invalidate_trainable_precision_caches
  invalidate_trainable_precision_caches(model)
  actual_loss,_=original(model,optimizer,sample,*args,**kwargs)
  for p,g in zip(model.head.parameters(),expected_grads):
   if g is not None: torch.testing.assert_close(p.grad,g,atol=3e-6,rtol=3e-4)
  assert abs(actual_loss-expected_loss)<max(1e-6,abs(expected_loss)*2e-4),(actual_loss,expected_loss)
  for k,v in expected.items(): torch.testing.assert_close(model.head.state_dict()[k],v,atol=3e-6,rtol=3e-4)
  for k,v in expected_opt['state'].items():
   for name,t in v.items():
    if torch.is_tensor(t): torch.testing.assert_close(optimizer.state_dict()['state'][k][name],t,atol=3e-6,rtol=3e-4)
  print(json.dumps(dict(event='real_two_rank_equivalence_pass',rank=int(os.environ['RANK']),loss=actual_loss,peak_memory_bytes=torch.cuda.max_memory_allocated())),flush=True)
  model.head.load_state_dict(head);optimizer.load_state_dict(state);invalidate_trainable_precision_caches(model)
  dt._completed_updates=0
 return original(model,optimizer,batch,*args,**kwargs)
dt._distributed_consistency_update=verify
dt.main()
