#pragma once

#include <vector>
#include <functional>

template <class D, class R>
class Function {
private:
    std::function<std::vector<R>(const std::vector<D>&)> function;
public:
    Function(const std::function<std::vector<R>(const std::vector<D>&)>& function);
    std::vector<R> operator () (const std::vector<D>& parameters) const;
};

template <class D, class R>
Function<D, R>::Function(const std::function<std::vector<R>(const std::vector<D>&)>& function) {
    if (!function) {
        throw std::runtime_error("Empty function");
    }

    this->function = function;
}

template <class D, class R>
std::vector<R> Function<D, R>::operator () (const std::vector<D>& parameters) const {
    return this->function(parameters);
}