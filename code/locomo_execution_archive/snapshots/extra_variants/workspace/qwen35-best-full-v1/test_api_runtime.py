import copy
import math
import unittest
from api_runtime import validate_completion,target_likelihood


class MeteringTests(unittest.TestCase):
    def sample(self):
        return {'usage':{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7},'choices':[
          {'index':1,'text':'B','finish_reason':'stop','prompt_token_ids':[8,9], 'token_ids':[4]},
          {'index':0,'text':'A','finish_reason':'stop','prompt_token_ids':[1,2,3], 'token_ids':[5]}]}

    def test_batch_order_and_metered_counts(self):
        choices=validate_completion(self.sample(),[[1,2,3],[8,9]])
        self.assertEqual([c['text'] for c in choices],['A','B'])

    def test_silent_prompt_change_rejected(self):
        value=self.sample();value['choices'][0]['prompt_token_ids']=[8,10]
        with self.assertRaises(ValueError):validate_completion(value,[[1,2,3],[8,9]])

    def test_invalid_and_missing_usage_rejected(self):
        for field in ('prompt_tokens','completion_tokens','total_tokens'):
            value=self.sample();value['usage'][field]+=1
            with self.assertRaises(AssertionError):validate_completion(value,[[1,2,3],[8,9]])
        value=self.sample();value.pop('usage')
        with self.assertRaises(ValueError):validate_completion(value,[[1,2,3],[8,9]])

    def test_target_probability_is_actual_target_not_top_token(self):
        choice={'prompt_token_ids':[1,2,30,40], 'prompt_logprobs':[None,{},
                 {'30':{'logprob':-2.0},'5':{'logprob':-.1}}, {'40':{'logprob':-4.0}}]}
        value=target_likelihood(choice,2,[30,40])
        self.assertEqual(value['mean_logprob'],-3)
        self.assertEqual(value['answer_tokens'],2)
        choice['prompt_logprobs'][2].pop('30')
        with self.assertRaises(ValueError):target_likelihood(choice,2,[30,40])

    def test_missing_positions_and_nonfinite_probability_rejected(self):
        value=self.sample()
        with self.assertRaises(ValueError):validate_completion(value,[[1,2,3],[8,9]],True)
        choice={'prompt_token_ids':[1,30], 'prompt_logprobs':[None,{'30':{'logprob':math.nan}}]}
        with self.assertRaises(ValueError):target_likelihood(choice,1,[30])


if __name__=='__main__':unittest.main()
