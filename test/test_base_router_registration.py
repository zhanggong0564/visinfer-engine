from routers.base_router import BaseRouter


class BatchRouter(BaseRouter):
    def request_schema(self, json_dict):
        return json_dict

    def get_inputs(self, request_params, image):
        return request_params, image


def test_base_router_can_skip_single_image_route_for_batch_scenes():
    router = BatchRouter(
        router_name="batch",
        api_path="/unused",
        summary="unused",
        description="unused",
        detector_type="batch",
        register_default_route=False,
    )

    assert router.get_router().routes == []
