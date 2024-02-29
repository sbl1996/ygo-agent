#ifndef MCTS_CORE_MINIMAX_H_
#define MCTS_CORE_MINIMAX_H_

#include <iostream>
#include <vector>

const float FLOAT_MAX = 1000000.0;
const float FLOAT_MIN = -FLOAT_MAX;

class MinMaxStats {
    public:
        float maximum, minimum, value_delta_max;

        MinMaxStats() {
            this->maximum = FLOAT_MIN;
            this->minimum = FLOAT_MAX;
            this->value_delta_max = 0.;
        }
        ~MinMaxStats() {}

        void set_delta(float value_delta_max) {
            this->value_delta_max = value_delta_max;
        }
        void update(float value) {
            if(value > this->maximum){
                this->maximum = value;
            }
            if(value < this->minimum){
                this->minimum = value;
            }
        }
        void clear() {
            this->maximum = FLOAT_MIN;
            this->minimum = FLOAT_MAX;
        }
        float normalize(float value) const {
            float norm_value = value;
            float delta = this->maximum - this->minimum;
            if(delta > 0){
                if(delta < this->value_delta_max){
                    norm_value = (norm_value - this->minimum) / this->value_delta_max;
                }
                else{
                    norm_value = (norm_value - this->minimum) / delta;
                }
            }
            return norm_value;            
        }
};

class MinMaxStatsList {
    public:
        int num;
        std::vector<MinMaxStats> stats_lst;

        MinMaxStatsList() {
            this->num = 0;
        }
        MinMaxStatsList(int num) {
            this->num = num;
            for(int i = 0; i < num; ++i){
                this->stats_lst.push_back(MinMaxStats());
            }
        }
        ~MinMaxStatsList() {}

        void set_delta(float value_delta_max) {
            for(int i = 0; i < this->num; ++i){
                this->stats_lst[i].set_delta(value_delta_max);
            }
        }
};

#endif // MCTS_CORE_MINIMAX_H_