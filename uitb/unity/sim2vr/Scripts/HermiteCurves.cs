using UnityEngine;

namespace UserInTheBox
{
    public class HermiteCurve
    {
        private float _xmin, _xeps, _xref, _xmax;
        private float _ymin, _yeps, _yref, _ymax;
        private float _dydxmin, _dydxeps, _dydxref, _dydxmax;
        private bool _positiveOnly, _clipToMaxDist;

        private float _x0, _x1, _p0, _p1, _m0, _m1;
        private float _t, _h00, _h10, _h01, _h11;

        public void Initialise(float maxDist=1.0f, float minReward=-1.0f, float epsDist=0.01f, float epsReward=0.0f, float refDist=0.5f, 
            bool positiveOnly=false, bool clipToMaxDist=true)
        {
            _xmax = maxDist;  //initial/maximum distance that can be typically reached; used to scale entire reward term relative to other rewards
            _ymax = minReward;  //minimum negative reward; used to scale entire reward term relative to other rewards
            _xeps = epsDist;  //"sufficient" distance to fulfill the pointing task (typically, this corresponds to the target radius); often, if dist<=_xeps (for the first time), an additional bonus reward is given
            _yeps = epsReward;  //reward given at target boundary (WARNING: needs to be non-positive!); should be chosen between 10%*_ymax and _ymin=0            _xref = refDist;  //"expected" distance; used to scale gradients of this reward term appropriately
            _xref = refDist;  //"expected" distance; used to scale gradients of this reward term appropriately
            _positiveOnly = positiveOnly;  //whether to ensure that all reward terms are non-negative (WARNING: non-negative values only guaranteed if initial distance _xmax is the maximum reachable distance!)
            _clipToMaxDist = clipToMaxDist;  //whether to return the same reward for all distances larger than maxDist (WARNING: if set to false, unexpected low rewards might be given!)
            
            _xmin = 0.0f;  //minimum distance
            _ymin = 0.0f;  //reward at target
            _dydxmin = 0.0f;  //reward gradient at target
            
            if ((_yeps < 0.1f*_ymax) | (_yeps > _ymin))
            {
                throw new System.ArgumentException("'_yeps' should be chosen between 10%*_ymax=" + 0.1f*_ymax + " and _ymin=" + _ymin + ", but was set to _yeps=" + _yeps + ".");
            }

            _dydxeps = (_yeps-_ymin)/(_xeps-_xmin);

            _dydxmax = 0.1f*(_ymax-_ymin)/(_xmax-_xmin);  //initial reward gradient

            var _alpha = (_xref-_xeps)/(_xmax-_xeps);  //0.5  //scale factor for linear interpolation between left secant (alpha=1) and right secant (alpha=0) -> if _xref is closer to _xeps, we need a more shallow gradient (i.e., alpha->0), if _xref is closer to _xmax, we need a steeper gradient (i.e., alpha->1) 

            _yref = 0.5f*(_ymin + _ymax);  //"expected"/"average" reward
            _dydxref = _alpha*(_yref-_yeps)/(_xref-_xeps) + (1-_alpha)*(_ymax-_yref)/(_xmax-_xref);  //-0.5  #"expected"/"average" reward gradient

            if (_positiveOnly)
            {
                _ymin -= _ymax;
                _yeps -= _ymax;
                _yref -= _ymax;
                _ymax -= _ymax;
            }
        }

        public float Evaluate(float x)
        {
            if (x < _xmin)
            {
                throw new System.ArgumentException("Invalid distance value x=" + x + ". Only values within [" + _xmin + ", " + _xmax + "] are accepted.");
            }
            else if (x < _xeps)
            {
                _x0 = _xmin;
                _x1 = _xeps;
                _p0 = _ymin;
                _p1 = _yeps;
                _m0 = _dydxmin;
                _m1 = _dydxeps;
            }
            else if (x < _xref)
            {
                _x0 = _xeps;
                _x1 = _xref;
                _p0 = _yeps;
                _p1 = _yref;
                _m0 = _dydxeps;
                _m1 = _dydxref;
            }
            else if (x <= _xmax)
            {
                _x0 = _xref;
                _x1 = _xmax;
                _p0 = _yref;
                _p1 = _ymax;
                _m0 = _dydxref;
                _m1 = _dydxmax;
            }
            else {
                // throw new System.ArgumentException("Invalid distance value x=" + x + ". Only values within [" + _xmin + ", " + _xmax + "] are accepted.");
                _x0 = _xref;
                _x1 = _xmax;
                _p0 = _yref;
                _p1 = _ymax;
                _m0 = _dydxref;
                _m1 = _dydxmax;

                if (_clipToMaxDist)
                {
                    x = _xmax;
                }
            }

            _t = (x - _x0) / (_x1 - _x0);
            _h00 = 2 * _t * _t * _t - 3 * _t * _t + 1;
            _h10 = _t * _t * _t - 2 * _t * _t + _t;
            _h01 = -2 * _t * _t * _t + 3 * _t * _t;
            _h11 = _t * _t * _t - _t * _t;

            // Console.WriteLine("_x0: " + _x0);
            // Console.WriteLine("_x1: " + _x1);
            // Console.WriteLine("_p0: " + _p0);
            // Console.WriteLine("_p1: " + _p1);
            // Console.WriteLine("_m0: " + _m0);
            // Console.WriteLine("_m1: " + _m1);

            return _h00 * _p0 + _h10 * (_x1 - _x0) * _m0 + _h01 * _p1 + _h11 * (_x1 - _x0) * _m1;
        }
    }
}