#ifndef DARKNET_HELPER_H
#define DARKNET_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

struct darknet_priv;

struct darknet_helper {
	struct darknet_priv *priv;
};

struct darknet_helper * darknet_init();
int darknet_load_network(struct darknet_helper *dnet, const char *cfg, const char *weights);
//void * darknet_image
//float * darknet_predict(struct darknet_helper *dnet, );

#ifdef __cplusplus
}
#endif

#endif // DARKNET_HELPER_H
