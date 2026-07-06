CC     = gcc
CFLAGS = -O2 -mavx2 -mfma -std=c11 -Wall -Wextra -I.
TARGET = neuroc47

SRCS = \
	source/structures/vector/vector.c \
	source/structures/matrix/matrix.c \
	source/derivatives/derivative.c \
	source/functions/loss_functions/mean_squared_error.c \
	source/optimizations/gradient_descent.c \
	source/network/forward_cache.c \
	source/network/layers/layer.c \
	source/network/network.c \
	main.c

OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	find . -name "*.o" -delete
	rm -f $(TARGET)

.PHONY: clean
