using UnityEngine;
 
public class InteractionManager : MonoBehaviour
{
    private BridgePart currentPart;
 
    void Update()
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;
 
        if (Physics.Raycast(ray, out hit))
        {
            BridgePart part = hit.collider.GetComponent<BridgePart>();
 
            if (part != null)
            {
                if (currentPart != part)
                {
                    if (currentPart != null)
                        currentPart.OnHoverExit();
 
                    currentPart = part;
                    currentPart.OnHoverEnter();
                }
 
                if (Input.GetMouseButtonDown(0))
                {
                    currentPart.OnClick();
                }
            }
        }
        else
        {
            if (currentPart != null)
            {
                currentPart.OnHoverExit();
                currentPart = null;
            }
        }
    }
}
 